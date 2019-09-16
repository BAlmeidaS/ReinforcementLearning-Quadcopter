import numpy as np
from itertools import product


class ActionSpace():
    DISCRETE_VALUES = [0, 200, 360, 404, 440, 700, 900]
    ACTIONS = np.array(list(product(*((DISCRETE_VALUES,)*4))))
    size = len(ACTIONS)

    @classmethod
    def get(cls, id):
        return cls.ACTIONS[id]

    @classmethod
    def sample(cls):
        return cls.get(np.random.randint(cls.size))
