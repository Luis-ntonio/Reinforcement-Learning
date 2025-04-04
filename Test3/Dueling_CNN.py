import tensorflow as tf
from tensorflow.keras import layers, models

class AssignmentCNNModel(tf.keras.Model):
    def __init__(self, rows, cols, product_feature_dim, matrix_channels, embed_dim=128):
        """
        Args:
          rows: Number of rows in the grid.
          cols: Number of columns in the grid.
          product_feature_dim: Dimension of each product's feature vector (should be 2: [ID, quantity]).
          matrix_channels: Number of channels for the matrix input (e.g., 1 if it's a weight map).
          embed_dim: The dimension of the learned embeddings.
          
        Assumes:
          num_products = rows * cols.
        """
        super(AssignmentCNNModel, self).__init__()
        self.rows = rows
        self.cols = cols
        self.num_products = rows * cols
        self.product_feature_dim = product_feature_dim  # should be 2
        self.matrix_channels = matrix_channels
        self.embed_dim = embed_dim
        print(f"Model initialized with rows: {rows}, cols: {cols}, product_feature_dim: {product_feature_dim}, matrix_channels: {matrix_channels}, embed_dim: {embed_dim}")
        
        # Process the matrix input with convolution layers.
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.cell_embedding = layers.Dense(embed_dim, activation='relu')
        
        # Process the product input with a dense layer.
        self.product_embedding = layers.Dense(embed_dim, activation='relu')
        
    def call(self, inputs):
        # Unpack inputs.
        # product_input shape: (batch, num_products, product_feature_dim)
        # matrix_input shape: (batch, rows, cols, matrix_channels)
        product_input, matrix_input = inputs
        batch_size = tf.shape(matrix_input)[0]
        
        # Process matrix input to get cell embeddings.
        x = self.conv1(matrix_input)
        x = self.conv2(x)
        x = self.conv3(x)
        cell_features = tf.reshape(x, (batch_size, self.rows * self.cols, -1))
        cell_emb = self.cell_embedding(cell_features)  # (batch, total_cells, embed_dim)
        
        # Process product input.
        prod_emb = self.product_embedding(product_input)  # (batch, num_products, embed_dim)
        
        # Compute scores.
        scores = tf.matmul(prod_emb, cell_emb, transpose_b=True)  # (batch, num_products, total_cells)
        assignment_probs = tf.nn.softmax(scores, axis=-1)  # probabilities for each product over cells

        
        # For training we won't use tf.argmax, but we output assignment_probs.
        # Return both assignment_probs and (a preliminary assignments grid via tf.argmax for debugging).
        chosen_cells = tf.argmax(assignment_probs, axis=-1)  # (batch, num_products)
        # Convert chosen_cells to grid coordinates.
        rows_idx = tf.math.floordiv(chosen_cells, self.cols)
        cols_idx = tf.math.mod(chosen_cells, self.cols)
        
        # Build scatter indices.
        batch_range = tf.reshape(tf.range(batch_size), (batch_size, 1))
        batch_range = tf.tile(batch_range, [1, self.num_products])
        batch_range = tf.expand_dims(batch_range, axis=-1)
        rows_idx = tf.cast(tf.expand_dims(rows_idx, axis=-1), tf.int32)
        cols_idx = tf.cast(tf.expand_dims(cols_idx, axis=-1), tf.int32)
        scatter_indices = tf.concat([batch_range, rows_idx, cols_idx], axis=-1)
        scatter_indices = tf.reshape(scatter_indices, (-1, 3))
        
        # Values: product_input (flattened), shape should be (batch*num_products, product_feature_dim)
        values = tf.reshape(product_input, (-1, self.product_feature_dim))
        output_shape = (batch_size, self.rows, self.cols, self.product_feature_dim)
        preliminary_grid = tf.scatter_nd(scatter_indices, values, output_shape)
        
        return preliminary_grid, assignment_probs
