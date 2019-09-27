from collections import deque
from task import Task
import numpy as np

from agents.DDPG.score_eval import ScoreEvaluator, print_iteaction

import os
from agents.DDPG.agent import DDPG
import json

from uuid import uuid4
import ray

from itertools import product

import tensorflow as tf

init_pose = np.array([5., 5., 5., 0, 0., 0.])
init_velocities = np.array([0., 0., 0.])
init_angle_velocities = np.array([0., 0., 0.])
runtime = 5.
target_pos = np.array([10., 10., 30.])

task = Task(init_pose,
            init_velocities,
            init_angle_velocities,
            runtime,
            target_pos)

best_score = -np.inf
score_eval = ScoreEvaluator(1)


def train_a_batch(uuid, reg=1, num_nodes=100, gamma=0.99, lr=1e-4,
                  tau=1e-3, no_mu=0, no_theta=.15, no_sigma=.3):

    # garantee reproducibility
    np.random.seed(37)
    tf.set_random_seed(43)

    d = {'reg': reg,
         'num_nodes': num_nodes,
         'gamma': gamma,
         'lr': lr,
         'tau': tau,
         'no_mu': no_mu,
         'no_theta': no_theta,
         'no_sigma': no_sigma}

    agent = DDPG(task, **d)

    folder = f'weights/{uuid}/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    num_episodes = 2000

    for i in range(num_episodes):
        state = task.reset()
        score = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)

            agent.step(action, reward, state, done)

            score += reward
            if done:
                score_eval.add(score)
                print_iteaction(i, score_eval)
                break

    agent.save_weights(folder + 'weights')
    score_eval.plot_avg_scores(folder + 'performace')

    with open(folder + "params.json", "w") as f:
        f.write(json.dumps({k: float(v) for k, v in d.items()},
                           indent=2))


@ray.remote
def grid_search(x, total):
    regs = np.array([1, .9, 1e-1, 1e-2])
    num_nodess = np.array([64, 128, 256, 512, 32])
    gammas = np.array([0.99])
    lrs = np.array([1e-3, 1e-4, 1e-5])
    taus = np.array([1e-3, 1e-4])
    no_mus = np.array([0])
    no_thetas = np.array([.15])
    no_sigmas = np.array([.3])

    # linear product of each variable
    all_objs = list(product(regs,
                            num_nodess,
                            gammas,
                            lrs,
                            taus,
                            no_mus,
                            no_thetas,
                            no_sigmas))

    fator = int(len(all_objs)/total)

    if x == 0:
        vrs = all_objs[:fator]
    elif x == (total-1):
        vrs = all_objs[x*fator:]
    else:
        vrs = all_objs[x*fator: (x+1)*fator]

    # looping for each
    for (reg, num_nodes, gamma, lr, tau, no_mu, no_theta,
         no_sigma) in vrs:
        train_a_batch(str(uuid4()),
                      reg=reg,
                      num_nodes=num_nodes,
                      gamma=gamma,
                      lr=lr,
                      tau=tau,
                      no_mu=no_mu,
                      no_theta=no_theta,
                      no_sigma=no_sigma)


def parallel():
    ray.init()
    try:
        futures = [grid_search.remote(i, 8) for i in range(8)]
        ray.get(futures)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    parallel()
