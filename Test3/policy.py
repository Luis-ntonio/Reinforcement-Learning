import numpy as np
import tensorflow as tf


def softmax_policy(estimator, observation, temperature=1.0):
    # Get Q-values and convert them to float32 for stability
    q_values = estimator(tf.expand_dims(observation, axis=0))[0].numpy().astype(np.float32)

    #replace NaN to 0
    q_values = np.nan_to_num(q_values, nan=0.0)

    # Avoid division by extremely small temperatures
    temperature = max(temperature, 1e-6)
    
    # Scale and shift the q_values for numerical stability
    q_scaled = q_values / temperature
    q_scaled -= np.max(q_scaled)
    
    # Compute exponentials
    exp_q = np.exp(q_scaled)
    sum_exp_q = np.sum(exp_q)
    
    # Check for potential issues: if sum_exp_q is 0 or NaN, use a uniform distribution
    if np.isnan(sum_exp_q) or sum_exp_q == 0:
        probs = np.ones_like(exp_q) / len(exp_q)
    else:
        probs = exp_q / sum_exp_q
    
    # Force normalization (ensure that the probabilities sum to exactly 1)
    probs = probs / np.sum(probs)
    return probs




# ------------------------------
# Update Target Network
# ------------------------------
def update_target_network(q_network, target_q_network):
    target_q_network.set_weights(q_network.get_weights())
