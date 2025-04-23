import tensorflow as tf
from tensorflow.keras import layers

class CNNPlacementQNetwork(tf.keras.Model):
    def __init__(self, grid_shape=(14, 7, 1), product_dim=5, output_dim=98):
        super().__init__()

        self.grid_conv = tf.keras.Sequential([
            layers.Conv2D(16, kernel_size=3, activation='relu', padding='same', input_shape=grid_shape),
            layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
            layers.Flatten()
        ])

        self.product_fc = tf.keras.Sequential([
            layers.Input(shape=(product_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu')
        ])

        self.output_head = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(output_dim)
        ])

    def call(self, inputs, training=False):
        grid_input, product_input = inputs
        grid_feat = self.grid_conv(grid_input, training=training)
        prod_feat = self.product_fc(product_input, training=training)
        combined = tf.concat([grid_feat, prod_feat], axis=-1)
        q_values = self.output_head(combined, training=training)
        return q_values
