import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def add(self, priority, data):
        tree_index = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_index, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def _propagate(self, tree_index, change):
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, value):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_index = parent
                break
            else:
                if value <= self.tree[left]:
                    parent = left
                else:
                    value -= self.tree[left]
                    parent = right
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5
        self.capacity = capacity

    def add(self, error, sample):
        priority = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, sample)

    def sample(self, batch_size, beta=0.4):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get(value)
            batch.append(data)
            indices.append(index)
            priorities.append(priority)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        is_weights = (self.capacity * sampling_probabilities) ** (-beta)
        is_weights /= is_weights.max()

        return batch, indices, is_weights

    def update_priorities(self, indices, errors):
        for index, error in zip(indices, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(index, priority)
