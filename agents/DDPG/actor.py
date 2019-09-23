import numpy as np

from keras import layers, models, optimizers
from keras import backend as K
import keras


class Actor():
    def __init__(self, env, lr=0.0001, tau=0.001):
        self.env = env

        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.shape
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low
        self.action_range = self.action_high - self.action_low

        self.lr = lr
        self.tau = tau

        self.model, output_layer = self.build_network()
        self.target_model, _ = self.build_network()

        # Define Loss
        action_gradients = layers.Input(shape=self.action_size)
        loss = K.mean(-action_gradients*output_layer)

        # Get trainable parameters and define backprop optimization.
        adam_optimizer = optimizers.Adam(lr=self.lr)
        train_param = adam_optimizer.get_updates(params=self.model.trainable_weights,
                                                 loss=loss)

        # keras.backend.learning_phase() gives a flag to be passed as input
        # to any Keras function that uses a different behavior at train time and test time.
        self.train_fn = K.function(inputs=[self.model.input,
                                           action_gradients,
                                           K.learning_phase()],
                                   outputs=[],
                                   updates=train_param)

    def build_network(self):
        states = layers.Input(shape=self.state_size, name='states')

        layer_1 = layers.Dense(
            units=256,
            activation='relu'
        )(states)
        layer_1 = layers.GaussianNoise(1.0)(layer_1)

        layer_2 = layers.Dense(
            units=128,
            activation='relu'
        )(layer_1)
        layer_2 = layers.GaussianNoise(1.0)(layer_2)

        output = layers.Dense(self.action_size[0],
                              activation='tanh',
                              kernel_initializer=keras.initializers.RandomUniform())(layer_2)
        output = layers.Lambda(lambda i: i * self.action_range)(output)

        return models.Model(inputs=[states], outputs=[output]), output

    def predict(self, state):
        prediction = self.model.predict(np.expand_dims(state, axis=0))

        return prediction[0]

    def transfer_learning(self):
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)