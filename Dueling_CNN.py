import tensorflow as tf
from tensorflow.keras import layers, models

class AssignmentCNNModel(tf.keras.Model):
    def __init__(self, rows, cols, product_feature_dim, matrix_channels, embed_dim=128, dropout_rate=0.8):
        super(AssignmentCNNModel, self).__init__()
        self.rows = rows
        self.cols = cols
        self.num_products = rows * cols
        self.product_feature_dim = product_feature_dim
        self.embed_dim = embed_dim

        # CNN layers for matrix input
        self.cnn_layers = models.Sequential([
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(dropout_rate),

            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(dropout_rate),

            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(dropout_rate),
        ])

        self.cell_embedding = layers.Dense(embed_dim, activation=None)

        # Product embedding
        self.product_embedding = models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim, activation='relu')
        ])

    def call(self, inputs, training=False):
        if len(inputs) == 3:
            product_input, matrix_input, product_mask = inputs
        else:
            product_input, matrix_input = inputs
            product_mask = tf.ones((tf.shape(product_input)[0], tf.shape(product_input)[1]), dtype=tf.float32)

        batch_size = tf.shape(matrix_input)[0]

        # Process matrix input
        x = self.cnn_layers(matrix_input, training=training)
        cell_features = tf.reshape(x, (batch_size, self.rows * self.cols, -1))
        cell_emb = self.cell_embedding(cell_features)  # (B, C, D)

        # Process product input
        prod_emb = self.product_embedding(product_input, training=training)  # (B, P, D)

        # Dot product for Q-values
        q_values = tf.matmul(prod_emb, cell_emb, transpose_b=True)  # (B, P, C)

        # Mask out padded products
        mask_expanded = tf.expand_dims(product_mask, axis=-1)  # (B, P, 1)
        q_values = q_values * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        return None, q_values
