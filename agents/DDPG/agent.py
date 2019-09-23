import numpy as np

from .actor import Actor
from .critic import Critic
from .utils import ReplayBuffer, OUNoise


class DDPG:
    def __init__(self, env, buffer_size=20000, batch_size=96, reg=1e-2,
                 gamma=0.99, lr=0.00005, tau=0.001,
                 no_mu=.1, no_theta=.2, no_sigma=0.25):
        """ Initialization
        """
        # Environment and A2C parameters

        self.state_size = (env.state_size,)
        self.action_size = (env.action_size,)
        self.action_high = env.action_high
        self.action_low = env.action_low
        self.action_range = self.action_high - self.action_low

        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(env, 0.1 * lr, tau, reg)
        self.critic = Critic(env, lr, tau, reg)

        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size, batch_size)

        self.noise = OUNoise(self.action_size,
                             no_mu,
                             no_theta,
                             no_sigma)

    def reset_episode(self):
        self.noise.reset()

    def act(self, state):
        self.last_state = state
        return self.actor.predict(state) + self.noise.sample()

    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state,
                        action,
                        reward,
                        next_state,
                        done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def bellman(self, rewards, q_values, dones):
        critic_targets = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_targets[i] = rewards[i]
            else:
                critic_targets[i] = rewards[i] + self.gamma * q_values[i]
        return critic_targets

    def memorize(self, state, action, reward, done, new_state):
        self.memory.add(state, action, reward, done, new_state)

    def sample_batch(self):
        return self.buffer.sample()

    def learn(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = (np.array([e.action for e in experiences if e is not None])
                     .astype(np.float32))
        rewards = (np.array([e.reward for e in experiences if e is not None])
                     .astype(np.float32))
        dones = (np.array([e.done for e in experiences if e is not None])
                   .astype(np.uint8))
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        q_values = (self.critic
                        .target_model
                        .predict([next_states,
                                  self.actor
                                      .target_model
                                      .predict(next_states)]))

        targets = self.bellman(rewards, q_values, dones)

        self.critic.train_on_batch(states, actions, targets)

        actions = self.actor.model.predict(states)
        gradients = self.critic.action_gradients(states, actions)

        self.actor.train_fn([states, gradients, 1])

        self.actor.transfer_learning()
        self.critic.transfer_learning()
