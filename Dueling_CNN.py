import tensorflow as tf
from tensorflow.keras import layers, models

class AssignmentCNNModel(tf.keras.Model):
    def __init__(self, rows, cols, product_feature_dim, matrix_channels, embed_dim=128, dropout_rate=0.5):
        super(AssignmentCNNModel, self).__init__()
        self.rows = rows
        self.cols = cols
        self.num_products = rows * cols
        self.product_feature_dim = product_feature_dim
        self.matrix_channels = matrix_channels
        self.embed_dim = embed_dim

        # CNN layers for matrix input
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()
        self.drop1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv2D(64, (3, 3), padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.ReLU()
        self.drop2 = layers.Dropout(dropout_rate)

        self.conv3 = layers.Conv2D(128, (3, 3), padding='same')
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.ReLU()
        self.drop3 = layers.Dropout(dropout_rate)

        self.cell_embedding = layers.Dense(embed_dim, activation='relu')

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

        # Matrix through CNN
        x = self.conv1(matrix_input)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.drop2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.drop3(x, training=training)

        cell_features = tf.reshape(x, (batch_size, self.rows * self.cols, -1))
        cell_emb = self.cell_embedding(cell_features)  # (batch, total_cells, embed_dim)

        # Product embedding
        prod_emb = self.product_embedding(product_input, training=training)

        # Attention scores
        scores = tf.matmul(prod_emb, cell_emb, transpose_b=True)

        # Apply mask to ignore padded products
        mask_expanded = tf.expand_dims(product_mask, axis=-1)
        scores = scores * mask_expanded + (1 - mask_expanded) * (-1e9)

        assignment_probs = tf.nn.softmax(scores, axis=-1)

        # For debugging/visualization (optional)
        chosen_cells = tf.argmax(assignment_probs, axis=-1)
        rows_idx = tf.math.floordiv(chosen_cells, self.cols)
        cols_idx = tf.math.mod(chosen_cells, self.cols)

        batch_range = tf.reshape(tf.range(batch_size), (batch_size, 1))
        batch_range = tf.tile(batch_range, [1, self.num_products])
        batch_range = tf.expand_dims(batch_range, axis=-1)

        rows_idx = tf.cast(tf.expand_dims(rows_idx, axis=-1), tf.int32)
        cols_idx = tf.cast(tf.expand_dims(cols_idx, axis=-1), tf.int32)
        scatter_indices = tf.concat([batch_range, rows_idx, cols_idx], axis=-1)
        scatter_indices = tf.reshape(scatter_indices, (-1, 3))

        values = tf.reshape(product_input, (-1, self.product_feature_dim))
        output_shape = (batch_size, self.rows, self.cols, self.product_feature_dim)
        preliminary_grid = tf.scatter_nd(scatter_indices, values, output_shape)

        return preliminary_grid, assignment_probs
