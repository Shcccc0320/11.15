# test_v2x_environment.py

import numpy as np
from V2XEnvironment import V2XEnvironment


def test_v2x_environment():
    # 创建环境实例
    env = V2XEnvironment()

    # 重置环境，获取初始状态
    state = env.reset()

    done = False
    total_reward = 0
    step = 0

    # 运行一个回合，直到所有车辆到达目的地
    while not done:
        # 从动作空间中随机采样一个动作
        action = env.action_space.sample()

        # 执行动作，获取下一个状态、奖励、是否结束和额外信息
        next_state, reward, done, info = env.step(action)

        # 累计总奖励
        total_reward += reward
        step += 1

        # 打印当前步的信息
        print(f"Step: {step}")
        print(f"Action taken: {action}")
        print(f"Reward received: {reward}")
        print(f"Total reward: {total_reward}")
        print(f"Done: {done}")
        print("-" * 50)

        # 更新状态
        state = next_state

        # 如果需要，可以渲染环境（注意：渲染会弹出窗口，需要关闭才能继续）
        # env.render()

    print(f"Episode finished after {step} steps")
    print(f"Total reward obtained: {total_reward}")

    # 关闭环境
    env.close()


if __name__ == "__main__":
    test_v2x_environment()
