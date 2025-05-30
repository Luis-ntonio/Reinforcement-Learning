import tensorflow as tf

def create_improved_model():
    """Create a more stable CNN model with batch normalization"""
    # Input layers
    grid_input = tf.keras.layers.Input(shape=(7, 14, 1))
    product_input = tf.keras.layers.Input(shape=(6,))
    
    # Grid processing branch with batch normalization
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(grid_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    grid_features = tf.keras.layers.Flatten()(x)
    
    # Product feature processing branch
    y = tf.keras.layers.Dense(32)(product_input)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Dense(64)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    product_features = tf.keras.layers.Activation('relu')(y)
    
    # Combined processing
    combined = tf.keras.layers.Concatenate()([grid_features, product_features])
    z = tf.keras.layers.Dense(256)(combined)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation('relu')(z)
    z = tf.keras.layers.Dense(128)(z)
    z = tf.keras.layers.Activation('relu')(z)
    output = tf.keras.layers.Dense(98)(z)  # No activation for Q-values
    
    # Create model
    model = tf.keras.Model(inputs=[grid_input, product_input], outputs=output)
    return model