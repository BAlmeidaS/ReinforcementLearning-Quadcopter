import numpy as np
import pandas as pd
from collections import deque

import matplotlib.pyplot as plt


class ScoreEvaluator:
    def __init__(self, window: int):
        self.window = window
        self.best_score = -np.inf
        self.avg_scores = []
        self.tmp_scores = deque(maxlen=window)
        self.last_score = None

    def add(self, score: float):
        if not score:
            raise ValueError(f'Score could not be {score}')

        self.tmp_scores.append(score)

        if score > self.best_score:
            self.best_score = score

        self._update_avg()
        self.last_score = score

    def plot_avg_scores(self):
        plt.plot(np.linspace(0,
                             len(self.avg_scores),
                             len(self.avg_scores),
                             endpoint=False),
                 np.asarray(self.avg_scores))
        plt.title(f'Best Reward: {self.best_score:10.5f}')
        plt.xlabel('Episode Number')
        plt.ylabel(f'Average Actions made (Over Next {self.window} Episodes)')
        rolling_mean = (pd.Series(self.avg_scores)
                          .rolling(199)
                          .mean())

        plt.plot(rolling_mean)

        plt.show()

    def _update_avg(self):
        if len(self.tmp_scores) < self.tmp_scores.maxlen:
            return
        self.avg_scores.append(np.mean(self.tmp_scores))


def print_iteaction(iteraction: int, score_eval: ScoreEvaluator):
    "function responsible to print some infos each iteration"
    print("{:4d} - Best Score: {:5.2f} - {:5.2f}".format(iteraction,
                                                         score_eval.best_score,
                                                         score_eval.last_score),
          end="\r",
          flush=True)
