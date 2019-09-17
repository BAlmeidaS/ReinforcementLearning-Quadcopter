from .actions import ActionSpace

import numpy as np

from .utils import create_tilings, tile_encode


class QTable:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.q_table = np.zeros(shape=(state_size + (action_size,)))

        print("QTable(): size =", self.q_table.shape)


class TiledQTable:
    def __init__(self, low, high, tiling_specs, action_size):
        self.tilings = create_tilings(low, high, tiling_specs)
        self.state_sizes = [tuple(len(splits)+1 for splits in tiling_grid) for tiling_grid in self.tilings]
        self.action_size = action_size
        self.q_tables = [QTable(state_size, action_size) for state_size in self.state_sizes]
        print("TiledQTable(): no. of internal tables = ", len(self.q_tables))

    def get(self, state, action):
        indices = tile_encode(state, self.tilings)

        return np.mean([self.q_tables[i].q_table[ind + (action, )] for i, ind in enumerate(indices)])

    def update(self, state, action, value, alpha=0.1):
        indices = tile_encode(state, self.tilings)

        for i, q_table in zip(indices, self.q_tables):
            q_s = q_table.q_table[i + (action, )]
            q_table.q_table[i + (action, )] = alpha * value + (1.0 - alpha) * q_s


class AgentSarsamax():
    def __init__(self, task, gamma=0.99, alpha=0.1):
        # constraints
        low_boundaries = (-30, -30, 0, -1, -1, -1)
        high_boundaries = (30, 30, 50, 1, 1, 1)

        bins = tuple(np.ones(len(high_boundaries), dtype=int) * 10)

        offsets = np.array([(high - low) for high, low in zip(high_boundaries,
                                                              low_boundaries)]) / 40

        second_pos_offsets = tuple(offsets * 2)
        first_pos_offsets = tuple(offsets * 1)
        zero_offsets = tuple(offsets * 0)
        first_neg_offsets = tuple(offsets * -1)
        second_neg_offsets = tuple(offsets * -2)

        tiling_specs = [(bins, second_pos_offsets),
                        (bins, first_pos_offsets),
                        (bins, zero_offsets),
                        (bins, first_neg_offsets),
                        (bins, second_neg_offsets)]

        tiling_specs = [(bins, first_pos_offsets),
                        (bins, zero_offsets),
                        (bins, first_neg_offsets)]

        # tiling_specs = [(bins, zero_offsets)]

        self.actions = ActionSpace.size

        self.q_table = TiledQTable(low_boundaries,
                                   high_boundaries,
                                   tiling_specs,
                                   self.actions)
        self.task = task
        self.gamma = gamma
        self.alpha = alpha

        #scores
        self.best_score = -np.inf
        self.score = 0

    def act(self, raw_state, epsilon):
        state = self._preprocess_state(raw_state)

        self.last_state = state

        do_exploration = np.random.uniform(0, 1) < epsilon
        if do_exploration:
            return ActionSpace.sample()
        else:
            return ActionSpace.get(np.argmax(self._greedy_policy(state)))

    def step(self, raw_next_state, raw_action, reward, done):
        """
        This is an aplication of sarsamax
        """
        if self.last_state is None:
            raise RuntimeError('trying to reward an action without running the agent yet!')

        next_state = self._preprocess_state(raw_next_state)
        action = ActionSpace.find(raw_action)

        value = reward + self.gamma * max(self._greedy_policy(next_state))

        self.q_table.update(self.last_state, action, value, alpha=self.alpha)

        self.score += reward

        if done:
            if self.score > self.best_score:
                self.best_score = self.score

    def _preprocess_state(self, raw_state):
        last_state = raw_state[-6:]
        sin_of_angles = np.sin(last_state[-3:])
        return np.concatenate((last_state[:3], sin_of_angles))

    def _greedy_policy(self, state):
        return [self.q_table.get(state, a) for a in range(self.actions)]
