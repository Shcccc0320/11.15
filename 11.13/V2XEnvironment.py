import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import shap  # 导入 SHAP 库
from gym import spaces
import networkx as nx
import gym

# V2XEnvironment Class
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class V2XEnvironment(gym.Env):
    def __init__(self):
        super(V2XEnvironment, self).__init__()

        # Initialize grid road network
        self.G = nx.Graph()
        # Add nodes and positions for intersections
        self.G.add_nodes_from([
            ('A', {'position': (0, 0)}),  # Intersection nodes
            ('B', {'position': (0, 3)}),
            ('C', {'position': (3, 0)}),
            ('D', {'position': (3, 3)})
        ])
        # Add edges to represent roads
        self.G.add_edges_from([
            ('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')
        ])

        # Base station configuration (0 is MBS, 1-4 are SBS)
        self.base_stations = {
            0: {'position': (1.5, 1.5), 'total_bandwidth': 300, 'transmission_power': 120},  # MBS configuration
            1: {'position': (0, 0), 'total_bandwidth': 100, 'transmission_power': 60},  # SBS at A
            2: {'position': (0, 3), 'total_bandwidth': 100, 'transmission_power': 60},  # SBS at B
            3: {'position': (3, 0), 'total_bandwidth': 100, 'transmission_power': 60},  # SBS at C
            4: {'position': (3, 3), 'total_bandwidth': 100, 'transmission_power': 60}   # SBS at D
        }

        self.noise_power = 1e-13  # Noise power

        self.num_vehicles = 4  # Number of vehicles
        self.num_stations = 5  # Number of base stations

        # Total available bandwidth
        self.total_available_bandwidth = 50e6  # 50 MHz
        # QoS requirements and penalty factors
        self.max_delay = 0.01  # Maximum acceptable delay for URLLC (seconds)
        self.min_rate = 10  # Minimum acceptable data rate for eMBB (Mbps)

        # Normalization constants for state representation
        self.max_position = 3.0  # Maximum coordinate value
        self.max_data_requirement = 100.0  # Maximum data requirement
        self.max_bandwidth = 300.0  # Maximum bandwidth

        # Define observation space and action space
        vehicle_state_size = 2 + 1 + 1 + 3 + 1  # Position (x, y), data requirement, bandwidth allocation, slice type (one-hot), active flag
        base_station_state_size = self.num_stations * (1 + 1 + 1)  # Distance, channel gain, total bandwidth
        total_state_size = self.num_vehicles * (vehicle_state_size + base_station_state_size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(total_state_size,), dtype=np.float32)

        # Define continuous action space for both base station selection and bandwidth allocation
        action_dim = self.num_vehicles * (self.num_stations + 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

        # Initialize environment state
        self.state = None
        self.reset()

    def reset(self):
        # Initialize vehicle positions and states
        routes = [
            ['A', 'B', 'D'],  # Vehicle 0 route
            ['C', 'D', 'B'],  # Vehicle 1 route
            ['B', 'A', 'C'],  # Vehicle 2 route
            ['D', 'C', 'A'],  # Vehicle 3 route
        ]

        self.state = {
            'vehicles': {
                i: {
                    'route': routes[i],
                    'route_index': 0,
                    'current_edge': (routes[i][0], routes[i][1]),
                    'position': self.G.nodes[routes[i][0]]['position'],
                    'distance_on_edge': 0.0,
                    'edge_length': self.calculate_edge_length(routes[i][0], routes[i][1]),
                    'velocity': np.random.uniform(0.01, 0.05),
                    'slice_type': np.random.choice(['URLLC', 'eMBB', 'Both']),
                    'has_arrived': False,
                    'data_requirement': None,
                    'bandwidth_allocation': 0.0,
                    'base_station': None,
                }
                for i in range(self.num_vehicles)
            }
        }

        # Set data requirements
        for vehicle in self.state['vehicles'].values():
            slice_type = vehicle['slice_type']
            if slice_type == 'URLLC':
                vehicle['data_requirement'] = max(np.random.uniform(10, 80), 1e-3)
            elif slice_type == 'eMBB':
                vehicle['data_requirement'] = max(np.random.uniform(20, 100), 1e-3)
            elif slice_type == 'Both':
                vehicle['data_requirement'] = max(np.random.uniform(10, 100), 1e-3)
            vehicle['bandwidth_allocation'] = 0.0

        return self.get_state()

    def calculate_edge_length(self, node1, node2):
        pos1 = np.array(self.G.nodes[node1]['position'])
        pos2 = np.array(self.G.nodes[node2]['position'])
        return np.linalg.norm(pos2 - pos1)

    def get_state(self):
        # Get the current state representation of the environment
        state = []

        for vehicle in self.state.get('vehicles', {}).values():
            active_flag = 0.0 if vehicle.get('has_arrived', False) else 1.0
            vehicle_state = [
                (vehicle['position'][0] / self.max_position) * active_flag,
                (vehicle['position'][1] / self.max_position) * active_flag,
                (vehicle['data_requirement'] / self.max_data_requirement) * active_flag,
                (vehicle['bandwidth_allocation'] / self.max_bandwidth) * active_flag,
                (1.0 if vehicle['slice_type'] == 'URLLC' else 0.0) * active_flag,
                (1.0 if vehicle['slice_type'] == 'eMBB' else 0.0) * active_flag,
                (1.0 if vehicle['slice_type'] == 'Both' else 0.0) * active_flag,
                active_flag
            ]

            for base_station_id, base_station in self.base_stations.items():
                distance = np.linalg.norm(
                    np.array(vehicle['position']) - np.array(base_station['position'])) / self.max_position
                channel_gain = self.calculate_channel_gain(vehicle['position'], base_station['position'])
                base_station_state = [
                    distance * active_flag,
                    channel_gain * active_flag,
                    (base_station['total_bandwidth'] / self.max_bandwidth) * active_flag,
                ]
                vehicle_state.extend(base_station_state)
            state.extend(vehicle_state)

        state_array = np.array(state, dtype=np.float32)
        return state_array

    def calculate_channel_gain(self, vehicle_position, bs_position):
        # Calculate the channel gain between a vehicle and a base station
        distance = np.linalg.norm(np.array(vehicle_position) - np.array(bs_position))
        distance = max(distance, 1e-3)  # Avoid division by zero
        path_loss_exponent = 2.0  # Path loss exponent
        return (1 / distance) ** path_loss_exponent

    def step(self, action):
        # Execute one time step within the environment
        base_station_selection = action[:self.num_vehicles * self.num_stations]
        bandwidth_allocation = action[self.num_vehicles * self.num_stations:]

        # Clip and normalize actions
        base_station_selection = base_station_selection.reshape(self.num_vehicles, self.num_stations)
        base_station_selection = np.argmax(base_station_selection, axis=1)  # Choose the best base station for each vehicle
        bandwidth_allocation = np.clip(bandwidth_allocation, 0, 1)

        for i, vehicle in self.state['vehicles'].items():
            if vehicle.get('has_arrived', False):
                continue

            # Assign base station
            vehicle['base_station'] = base_station_selection[i]

            # Assign bandwidth
            vehicle['bandwidth_allocation'] = bandwidth_allocation[i] * self.base_stations[vehicle['base_station']]['total_bandwidth']

            # Move vehicle along its route
            move_distance = vehicle['velocity']
            vehicle['distance_on_edge'] += move_distance

            if vehicle['distance_on_edge'] >= vehicle['edge_length']:
                vehicle['route_index'] += 1
                if vehicle['route_index'] >= len(vehicle['route']) - 1:
                    vehicle['has_arrived'] = True
                    vehicle['position'] = self.G.nodes[vehicle['route'][-1]]['position']
                else:
                    from_node = vehicle['route'][vehicle['route_index']]
                    to_node = vehicle['route'][vehicle['route_index'] + 1]
                    vehicle['current_edge'] = (from_node, to_node)
                    vehicle['edge_length'] = self.calculate_edge_length(from_node, to_node)
                    vehicle['distance_on_edge'] = 0.0
            else:
                from_node, to_node = vehicle['current_edge']
                from_pos = np.array(self.G.nodes[from_node]['position'])
                to_pos = np.array(self.G.nodes[to_node]['position'])
                ratio = vehicle['distance_on_edge'] / vehicle['edge_length']
                vehicle['position'] = from_pos + (to_pos - from_pos) * ratio

        # Calculate reward
        reward = self.calculate_reward()

        # Get next state
        next_state = self.get_state()

        # Check if done
        done = all(vehicle.get('has_arrived', False) for vehicle in self.state['vehicles'].values())

        return next_state, reward, done, {}

    def calculate_reward(self):
        # Reward is based on QoS requirements and vehicle states
        reward = 0.0
        for vehicle in self.state['vehicles'].values():
            if vehicle.get('has_arrived', False):
                continue
            base_station_id = vehicle['base_station']
            slice_type = vehicle['slice_type']
            datarate = self.calculate_datarate(vehicle, base_station_id)
            delay = self.calculate_delay(vehicle, base_station_id)
            if slice_type == 'URLLC':
                reward += 2.0 if delay <= self.max_delay else -0.1 * (delay - self.max_delay)
            elif slice_type == 'eMBB':
                reward += 2.0 if datarate >= self.min_rate else -0.05 * (self.min_rate - datarate)
            elif slice_type == 'Both':
                if delay <= self.max_delay:
                    reward += 3
                else:
                    reward -= 0.1 * (delay - self.max_delay)

                if datarate >= self.min_rate:
                    reward += 3
                else:
                    reward -= 0.05 * (self.min_rate - datarate)

        return np.clip(reward, -10, 10)

    def calculate_datarate(self, vehicle, base_station_id):
        # Calculate the data rate for a vehicle connected to a base station
        W_i_m = vehicle['bandwidth_allocation']
        if W_i_m <= 1e-3 or base_station_id is None:
            return 0.0
        P = self.base_stations[base_station_id]['transmission_power']
        Gt_i_m = self.calculate_channel_gain(vehicle['position'], self.base_stations[base_station_id]['position'])
        sigma2 = self.noise_power
        sinr = (P * Gt_i_m) / (sigma2 + 0.1)
        datarate_bps = W_i_m * np.log2(1 + sinr)
        datarate_mbps = datarate_bps / 1e6
        return datarate_mbps if sinr > 0 else 0.0

    def calculate_delay(self, vehicle, base_station_id):
        # Calculate the delay for a vehicle based on data requirement and data rate
        data_size = vehicle['data_requirement']
        datarate_mbps = self.calculate_datarate(vehicle, base_station_id)
        return data_size / datarate_mbps if datarate_mbps > 1e-3 else 0.1

# Note: The rest of the DDPG agent and the training loop would be added here
# This environment can now be used directly with the provided DDPG implementation.


    def check_if_done(self):
        all_arrived = all(vehicle.get('has_arrived', False) for vehicle in self.state['vehicles'].values())
        return all_arrived

    def render(self, mode='human'):
        # 使用matplotlib渲染环境的当前状态
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        for bs_id, bs in self.base_stations.items():
            plt.scatter(bs['position'][0], bs['position'][1],
                        c='red' if bs_id == 0 else 'green',
                        marker='^' if bs_id == 0 else 'v', s=200)
            plt.text(bs['position'][0] + 0.05, bs['position'][1] + 0.05,
                     f'MBS' if bs_id == 0 else f'SBS{bs_id}')
        pos = nx.get_node_attributes(self.G, 'position')
        nx.draw(self.G, pos, node_color='gray', node_size=50, with_labels=True)
        for vehicle in self.state.get('vehicles', {}).values():
            if vehicle.get('has_arrived', False):
                continue
            plt.scatter(vehicle['position'][0], vehicle['position'][1], c='blue', s=100)
        plt.xlim(-1, 4)
        plt.ylim(-1, 4)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('V2X Environment')
        plt.show()
        plt.close()

    def close(self):
        # 关闭环境（这里不需要具体操作）
        pass
