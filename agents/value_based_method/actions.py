import numpy as np
from itertools import product


class ActionSpace():
    # DISCRETE_VALUES = [0, 200, 360, 404, 440, 700, 900]
    # ACTIONS = np.array(list(product(*((DISCRETE_VALUES,)*4))))
    ACTIONS = np.array([[404, 404, 404, 404],
                        [500, 500, 500, 500],
                        [300, 300, 300, 300],
                        [100, 100, 100, 100],
                        [500, 500, 404, 404],
                        [404, 500, 500, 404],
                        [404, 404, 500, 500],
                        [500, 404, 404, 500],
                        [500, 500, 450, 450],
                        [450, 500, 500, 450],
                        [450, 450, 500, 500],
                        [500, 450, 450, 500]])
    size = len(ACTIONS)

    @classmethod
    def get(cls, id):
        return cls.ACTIONS[id]

    @classmethod
    def find(cls, action):
        return np.where((cls.ACTIONS == action).all(axis=1))[0][0]

    @classmethod
    def sample(cls):
        return cls.get(np.random.randint(cls.size))
