# ActorCritic_CNN.py
import tensorflow as tf
from tensorflow.keras import layers

class ActorCriticCNN(tf.keras.Model):
    def __init__(self, rows, cols, product_feature_dim, matrix_channels, embed_dim=128):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.total_cells = rows * cols
        self.product_feature_dim = product_feature_dim

        # Matrix CNN embedding
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.cell_embed = layers.Dense(embed_dim, activation='relu')

        # Product embedding
        self.product_embed = layers.Dense(embed_dim, activation='relu')

        # Actor head: assignment probabilities
        self.actor_logits = layers.Dense(self.total_cells)

        # Critic head: state value
        self.critic_dense = layers.Dense(128, activation='relu')
        self.critic_value = layers.Dense(1)

    def call(self, inputs):
        product_input, matrix_input, product_mask = inputs
        batch_size = tf.shape(product_input)[0]

        # Matrix input → CNN → flatten + embed
        x = self.conv1(matrix_input)
        x = self.conv2(x)
        x = self.conv3(x)
        cell_features = tf.reshape(x, (batch_size, self.total_cells, -1))
        cell_emb = self.cell_embed(cell_features)

        # Product input → embed
        prod_emb = self.product_embed(product_input)

        # Actor: attention scores between products and cells
        # Softmax scores: (batch, num_products, total_cells)
        scores = tf.matmul(prod_emb, cell_emb, transpose_b=True)
        probs = tf.nn.softmax(scores, axis=-1)

        # Apply mask to remove padded products from the loss/gradients
        # product_mask: shape (batch, total_cells), we need (batch, num_products, 1)
        product_mask = tf.expand_dims(product_mask, axis=-1)
        probs = probs * product_mask  # zero-out probs from fake products

        # Critic: global state embedding → scalar value
        flat_matrix = tf.reshape(matrix_input, [batch_size, -1])
        flat_products = tf.reshape(product_input, [batch_size, -1])
        critic_input = tf.concat([flat_matrix, flat_products], axis=-1)
        value_hidden = self.critic_dense(critic_input)
        state_value = self.critic_value(value_hidden)  # (batch, 1)

        return probs, state_value
