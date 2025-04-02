import numpy as np
import tensorflow as tf

class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, sigma_init=0.017, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.sigma_init = sigma_init

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # Initialize mu and sigma for weights and biases in float16
        self.mu_weight = self.add_weight(
            name='mu_weight', 
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim)),
            trainable=True,
            dtype=tf.float16)
        self.sigma_weight = self.add_weight(
            name='sigma_weight', 
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.Constant(self.sigma_init),
            trainable=True,
            dtype=tf.float16)
        self.mu_bias = self.add_weight(
            name='mu_bias', 
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim)),
            trainable=True,
            dtype=tf.float16)
        self.sigma_bias = self.add_weight(
            name='sigma_bias', 
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.sigma_init),
            trainable=True,
            dtype=tf.float16)
        super(NoisyDense, self).build(input_shape)

    def call(self, inputs, training=True):
        # Ensure inputs are in float16
        inputs = tf.cast(inputs, tf.float16)
        if training:
            # Generate factorized Gaussian noise as float16
            input_noise = tf.random.normal((inputs.shape[-1],), dtype=tf.float16)
            output_noise = tf.random.normal((self.units,), dtype=tf.float16)
            f_input = tf.sign(input_noise) * tf.sqrt(tf.abs(input_noise))
            f_output = tf.sign(output_noise) * tf.sqrt(tf.abs(output_noise))
            noise_weight = tf.einsum('i,j->ij', f_input, f_output)
            noise_bias = f_output
            weight = self.mu_weight + self.sigma_weight * noise_weight
            bias = self.mu_bias + self.sigma_bias * noise_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        output = tf.matmul(inputs, weight) + bias
        if self.activation is not None:
            output = self.activation(output)
        return output
