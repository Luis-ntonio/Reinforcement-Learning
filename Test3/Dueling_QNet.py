from NoisyDense import NoisyDense
import tensorflow as tf

class DuelingQNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim, use_noisy=False):
        super(DuelingQNetwork, self).__init__()
        self.use_noisy = use_noisy
        # Common feature extractor layers
        if use_noisy:
            self.fc1 = NoisyDense(256, activation='relu')
            self.fc2 = NoisyDense(256, activation='relu')
        else:
            self.fc1 = tf.keras.layers.Dense(256, activation='relu')
            self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        # Dueling streams for value and advantage
        if use_noisy:
            self.value_fc = NoisyDense(128, activation='relu')
            self.value = NoisyDense(1, activation=None)
            self.advantage_fc = NoisyDense(128, activation='relu')
            self.advantage = NoisyDense(output_dim, activation=None)
        else:
            self.value_fc = tf.keras.layers.Dense(128, activation='relu')
            self.value = tf.keras.layers.Dense(1, activation=None)
            self.advantage_fc = tf.keras.layers.Dense(128, activation='relu')
            self.advantage = tf.keras.layers.Dense(output_dim, activation=None)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        # Value stream
        value = self.value_fc(x)
        value = self.value(value)
        # Advantage stream
        advantage = self.advantage_fc(x)
        advantage = self.advantage(advantage)
        # Combine streams: Q = V + (A - mean(A))
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        q_values = value + (advantage - advantage_mean)
        return q_values