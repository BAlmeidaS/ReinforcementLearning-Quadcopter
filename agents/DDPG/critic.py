import numpy as np

from keras import layers, models, optimizers
from keras import backend as K
import keras


class Critic():
    def __init__(self, env, lr=0.0001, tau=0.001):
        self.env = env

        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.shape
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low
        self.action_range = self.action_high - self.action_low

        self.lr = lr
        self.tau = tau

        self.model = self.build_network()
        self.target_model = self.build_network()

        self.grads = K.function([self.model.input[0],
                                 self.model.input[1]],
                                K.gradients(self.model.output,
                                            self.model.input[1]))

    def build_network(self):
        states = layers.Input(shape=self.state_size, name='states')
        actions = layers.Input(shape=self.action_size, name='actions')

        layer_1 = layers.Dense(
            units=256,
            activation='relu'
        )(states)

        layer_2 = layers.concatenate([layer_1, actions])

        layer_3 = layers.Dense(
            units=128,
            activation='relu'
        )(layer_2)

        output = layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer=keras.initializers.RandomUniform()
        )(layer_3)

        model = models.Model(inputs=[states, actions], outputs=output)

        optimizer = optimizers.Adam(lr=self.lr)

        model.compile(optimizer=optimizer, loss='mse')

        return model

    def action_gradients(self, states, actions):
        return self.grads([states, actions])[0]

    def predict(self, state, action):
        return self.model.predict([np.expand_dims(state, axis=0),
                                   np.expand_dims(action, axis=0)])[0]

    def train_on_batch(self, states, actions, critic_target):
        return self.model.train_on_batch([states, actions],
                                         critic_target)

    def transfer_learning(self):
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)
