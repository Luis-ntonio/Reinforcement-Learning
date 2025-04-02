from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from collections import namedtuple
import os
import logging
import pandas as pd
from NoisyDense import NoisyDense
from Transition import Transition, NStepTransitionBuffer
from policy import softmax_policy, update_target_network
from ReplayBuffer import PrioritizedReplayBuffer
from Assignment import AssignmentEnv
from Dueling_CNN import AssignmentCNNModel



gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled on all GPUs.")
    except RuntimeError as e:
        print(e)



def one_step_policy_learning(env, policy_model, num_episodes, learning_rate=0.0005):
    """
    Runs a one-step policy gradient training loop.
    Each episode, the model gets the current state (product features and weight matrix),
    produces an assignment (a grid of shape (rows, cols)), and the environment computes a penalty.
    The policy is updated using REINFORCE.
    """
    optimizer = Adam(learning_rate=learning_rate)
    rewards_list = []
    
    for episode in range(num_episodes):
        # Reset environment: get state as a tuple (product_features, matrix)
        state = env.reset()  # state = (product_features, matrix)
        product_features, matrix = state
        
        # Convert to float32 arrays.
        product_features = np.array(product_features, dtype=np.float32)  # shape: (num_products, product_feature_dim)
        matrix = np.array(matrix, dtype=np.float32)                      # shape: (rows, cols)
        # Add channel dimension to matrix.
        # matrix = matrix[..., np.newaxis]  # shape: (rows, cols, 1)
        
        # Add batch dimension to each input.
        product_features_batch = np.expand_dims(product_features, axis=0)  # (1, num_products, product_feature_dim)
        matrix_batch = np.expand_dims(matrix, axis=0)                      # (1, rows, cols, 1)
        state_batch = (product_features_batch, matrix_batch)
        
        with tf.GradientTape() as tape:
            # Pass the state through the policy model.
            # The model returns a grid and the assignment probabilities.
            # (For training we use the probabilities to sample an action.)
            assignments_grid, assignment_probs = policy_model(state_batch)
            # assignment_probs has shape: (1, num_products, total_cells)
            # Remove the batch dimension:
            assignment_probs = tf.squeeze(assignment_probs, axis=0)  # now shape: (num_products, total_cells)
            
            # Sample an action for each product from its probability distribution.
            # We use categorical sampling on the log probabilities.
            log_probs = tf.math.log(assignment_probs + 1e-8)  # shape: (num_products, total_cells)
            # For each product (row), sample one action (cell index).
            actions = tf.random.categorical(log_probs, num_samples=1)  # shape: (num_products, 1)
            actions = tf.squeeze(actions, axis=1)  # shape: (num_products,)
            
            # Gather the log probabilities of the chosen actions.
            # Create indices for each product.
            indices = tf.stack([tf.cast(tf.range(tf.shape(actions)[0]), tf.int64), actions], axis=1) # shape: (num_products, 2)
            chosen_log_probs = tf.gather_nd(log_probs, indices)  # shape: (num_products,)
            total_log_prob = tf.reduce_sum(chosen_log_probs)  # scalar

        # Convert sampled actions to a numpy array (this will be our action vector).
        action_vector = actions.numpy()  # shape: (num_products,)
        
        # Execute the action in the environment. The environment's step() returns:
        #   grid: the assignment grid (rows x cols) with product indices.
        #   reward: the penalization (or reward) computed from that assignment.
        #   done: True (one-step episode)
        grid, reward, done = env.step(action_vector)
        
        # Define the policy gradient loss as negative log-likelihood weighted by the reward.
        # (Assuming higher reward is better, so we minimize -log(prob)*reward.)
        loss = - total_log_prob * reward
        
        # Compute gradients and update the model.
        grads = tape.gradient(loss, policy_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))
        
        rewards_list.append(reward)
        print(f"Episode {episode+1}, Reward: {reward}, Loss: {loss.numpy()}")
    
    return rewards_list

# ------------------------------
# Main Execution Example
# ------------------------------
if __name__ == "__main__":
    import pandas as pd
    import os
    import logging

    # Logging configuration
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('log.txt')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load your data from Excel (as in your original code)
    file_path = 'productos_anaquel.xls'
    df_list = []
    i = 1
    try:
        while True:
            df_list.append(pd.read_excel(file_path, sheet_name=f"Sheet {i}"))
            i += 1
    except Exception as e:
        pass
    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all[df_all['ANAQUEL'].str.startswith('C', na=False)]
    df_all = df_all[df_all['CAMPA'] == 201416]
    df_all['UNDESTIMADAS'] = df_all['UNDESTIMADAS'].apply(lambda x: x if x > 0 else 1)
    df_all.reset_index(drop=True, inplace=True)
    df_all = df_all[['PRODUCTO', 'UNDESTIMADAS']]
    
    # Define feature dimension.
    feature_dim = 1  # Using 'UNDESTIMADAS' as the single product feature.
    # For simplicity, we assume that the number of products equals the total number of cells.
    # In the environment, the weight matrix defines the grid dimensions.
    # For example, if the weight matrix has shape (7, N), then:
    num_products = len(df_all)
    
    # Create the environment.
    from Assignment import AssignmentEnv
    env = AssignmentEnv(df_all)
    
    # Create the policy model.
    # IMPORTANT: The model constructor for AssignmentCNNModel is defined as:
    #   AssignmentCNNModel(rows, cols, product_feature_dim, matrix_channels, embed_dim=128)
    # Here, env.rows and env.cols come from the weight matrix.
    policy_model = AssignmentCNNModel(env.rows, env.cols, feature_dim, 1, embed_dim=128)
    
    # Train the model with one-step policy gradient.
    num_episodes = 1000
    rewards_list = one_step_policy_learning(env, policy_model, num_episodes, learning_rate=0.0005)
    
    # Save your model weights if needed.
    checkpoint_dir = os.path.join("./experiments", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model.weights.h5")
    policy_model.save_weights(checkpoint_path)