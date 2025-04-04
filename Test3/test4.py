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
from scipy.optimize import linear_sum_assignment


gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled on all GPUs.")
    except RuntimeError as e:
        print(e)

def unique_assignment(assignment_probs_np):
    """
    Given an assignment probability matrix of shape (num_products, total_cells),
    use the Hungarian algorithm to get a unique assignment.
    Returns an action vector of length num_products.
    """
    # We use negative probabilities as cost (to maximize probability).
    cost_matrix = -assignment_probs_np  # shape: (num_products, total_cells)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # linear_sum_assignment returns row indices and column indices.
    # We assume the cost matrix is square (num_products == total_cells).
    # If not, you need additional handling.
    # Return an action vector where for product i (assumed to be in sorted order),
    # the assigned cell is col_ind[i].
    # If row_ind is not sorted, sort them:
    sorted_indices = np.argsort(row_ind)
    action_vector = col_ind[sorted_indices]
    return action_vector


def one_step_policy_learning(env, policy_model, num_episodes, learning_rate=0.0005,
                             temperature_start=1.0, temperature_end=0.1, temperature_decay_steps=10000):
    optimizer = Adam(learning_rate=learning_rate)
    rewards_list = []
    global_step = 0
    for episode in range(num_episodes):
        # Get state from environment.
        state = env.reset()  # state = (product_features, matrix_input)
        product_features, matrix_input = state
        product_features = np.array(product_features, dtype=np.float32)  # shape: (num_products, 2)
        matrix_input = np.array(matrix_input, dtype=np.float32)          # shape: (rows, cols, 1)
        
        # Add batch dimension.
        product_features_batch = np.expand_dims(product_features, axis=0)  # (1, num_products, 2)
        matrix_batch = np.expand_dims(matrix_input, axis=0)                  # (1, rows, cols, 1)
        state_batch = (product_features_batch, matrix_batch)
        
        # Forward pass: get assignment probabilities.
        _, assignment_probs = policy_model(state_batch)
        # assignment_probs shape: (1, num_products, total_cells)
        assignment_probs_np = assignment_probs.numpy().squeeze(0)  # (num_products, total_cells)
        
        # Compute temperature (linearly decaying)
        temperature = max(temperature_end, temperature_start - (temperature_start - temperature_end) * global_step / temperature_decay_steps)
        
        # For each product, adjust the probabilities with temperature.
        # Here we simply divide the logits by temperature before re-applying softmax.
        # However, since our model already produced probabilities, we can simulate this by raising them to the power (1/temperature)
        # and then renormalizing.
        adjusted_probs = np.power(assignment_probs_np, 1.0/temperature)
        adjusted_probs /= np.sum(adjusted_probs, axis=-1, keepdims=True)
        
        # Use Hungarian algorithm to get unique assignment.
        action_vector = unique_assignment(adjusted_probs)  # action_vector shape: (num_products,)
        
        # For policy gradient, compute the log-probabilities of the chosen actions.
        chosen_log_probs = []
        for i in range(adjusted_probs.shape[0]):
            prob = adjusted_probs[i, action_vector[i]]
            chosen_log_probs.append(np.log(prob + 1e-8))
        total_log_prob = np.sum(chosen_log_probs)
        total_log_prob = tf.convert_to_tensor(total_log_prob, dtype=tf.float32)
        
        # Execute action in environment.
        grid, reward, done = env.step(action_vector)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        # REINFORCE loss: negative log-likelihood weighted by reward.
        loss = - total_log_prob * reward
        
        with tf.GradientTape() as tape:
            # Re-run forward pass for gradient computation.
            _, assignment_probs_grad = policy_model(state_batch)
            assignment_probs_grad = tf.squeeze(assignment_probs_grad, axis=0)  # (num_products, total_cells)
            # Gather the probabilities for the chosen actions.
            chosen_probs = []
            for i in range(assignment_probs_grad.shape[0]):
                chosen_probs.append(assignment_probs_grad[i, action_vector[i]])
            chosen_probs = tf.stack(chosen_probs)
            grad_log_prob = tf.reduce_sum(tf.math.log(chosen_probs + 1e-8))
            grad_loss = - grad_log_prob * reward
        grads = tape.gradient(grad_loss, policy_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))
        
        rewards_list.append(reward.numpy())
        print(f"Episode {episode+1}, Reward: {reward.numpy()}, Loss: {loss.numpy()}")
        global_step += 1
    return rewards_list

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    # Setup logging (optional)
    import os, logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('log.txt')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Load data from Excel (adapt as needed)
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
    # Ensure quantity is positive.
    df_all['UNDESTIMADAS'] = df_all['UNDESTIMADAS'].apply(lambda x: x if x > 0 else 1)
    df_all.reset_index(drop=True, inplace=True)
    df_all = df_all[['PRODUCTO', 'UNDESTIMADAS']]
    
    # Set product feature dimension to 2 ([ID, quantity]).
    feature_dim = 2
    # For this design, we'll pad/truncate products to match the number of grid cells defined by the weight matrix.
    # Create the environment.
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