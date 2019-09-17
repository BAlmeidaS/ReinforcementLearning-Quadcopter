import random

class Basic_Agent():
    def __init__(self, task):
        self.task = task

    def act(self):
        return [500.0, 550.0, 500.0, 550.0]
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]
