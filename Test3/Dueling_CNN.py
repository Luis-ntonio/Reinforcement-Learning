import tensorflow as tf
from tensorflow.keras import layers, models

class AssignmentCNNModel(tf.keras.Model):
    def __init__(self, rows, cols, product_feature_dim, matrix_channels, embed_dim=128):
        """
        Args:
          rows: Number of rows in the grid.
          cols: Number of columns in the grid.
          product_feature_dim: Dimension of each product's feature vector.
          matrix_channels: Number of channels for the matrix input (e.g., 1 if it’s a weight map).
          embed_dim: The dimension of the learned embeddings.
          
        Assumes:
          num_products = rows * cols.
        """
        super(AssignmentCNNModel, self).__init__()
        self.rows = rows
        self.cols = cols
        self.num_products = rows * cols
        self.product_feature_dim = product_feature_dim
        self.matrix_channels = matrix_channels
        self.embed_dim = embed_dim
        print(f"Model initialized with rows: {rows}, cols: {cols}, product_feature_dim: {product_feature_dim}, matrix_channels: {matrix_channels}, embed_dim: {embed_dim}")
        
        # Process the matrix input with a few convolutional layers.
        self.conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')
        self.conv3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')
        # We'll reshape the resulting feature map to obtain a "cell embedding" per cell.
        self.cell_embedding = layers.Dense(embed_dim, activation='relu')
        
        # Process the product input (each product's features) with a dense layer to get an embedding.
        self.product_embedding = layers.Dense(embed_dim, activation='relu')
        
    def call(self, inputs):
        # Unpack the two inputs.
        # product_input shape: (batch, num_products, product_feature_dim)
        # matrix_input shape: (batch, rows, cols, matrix_channels)
        product_input, matrix_input = inputs
        batch_size = tf.shape(matrix_input)[0]
        print(f"Matrix input shape: {matrix_input.shape}")
        
        # Process matrix input to get cell embeddings.
        # Convolve the matrix input.
        x = self.conv1(matrix_input)
        x = self.conv2(x)
        x = self.conv3(x)
        # Reshape the output to have one vector per cell.
        # Suppose the convolution outputs a tensor of shape (batch, rows, cols, F).
        # We reshape to (batch, rows*cols, F) where F is the number of features.
        cell_features = tf.reshape(x, (batch_size, self.rows * self.cols, -1))
        # Map cell features to embeddings.
        cell_emb = self.cell_embedding(cell_features)  # shape: (batch, total_cells, embed_dim)
        
        # Process product input to get product embeddings.
        prod_emb = self.product_embedding(product_input)  # shape: (batch, num_products, embed_dim)
        
        # Compute a score for each product–cell pair.
        # Use a dot product: we want a tensor of shape (batch, num_products, total_cells)
        # prod_emb: (batch, num_products, embed_dim)
        # cell_emb: (batch, embed_dim, total_cells)
        scores = tf.matmul(prod_emb, cell_emb, transpose_b=True)  # shape: (batch, num_products, total_cells)
        
        # For each product, turn scores into probabilities over cells.
        assignment_probs = tf.nn.softmax(scores, axis=-1)
        
        # Choose the best cell (highest probability) for each product.
        chosen_cells = tf.argmax(assignment_probs, axis=-1)  # shape: (batch, num_products)
        
        # Optionally, if you want the final output as a grid, reshape the chosen_cells:
        # Here we assume that the ordering of products corresponds to cells in row-major order.
        assignments_grid = tf.reshape(chosen_cells, (batch_size, self.rows, self.cols))
        return assignments_grid, assignment_probs

