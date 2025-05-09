import tensorflow as tf
from collections import deque
import random
import numpy as np

class EnhancedDQNAgent:
    def __init__(self, q_net, target_net, optimizer, 
                 gamma=0.99, 
                 replay_capacity=100000,
                 batch_size=64, 
                 update_freq=4,
                 tau=0.005,  # Increased for faster adaptation
                 reward_scale=0.005,  # Further reduced to prevent extreme Q-values
                 grad_clip=0.5,
                 prioritized_replay=True,
                 huber_loss=True,
                 n_step_returns=3):  # N-step returns for temporal difference
        
        self.q_net = q_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.tau = tau
        self.reward_scale = reward_scale
        self.prioritized_replay = prioritized_replay
        self.huber_loss = huber_loss
        self.n_step = n_step_returns
        
        # N-step returns buffer
        self.n_step_buffer = deque(maxlen=n_step_returns)
        
        # Initialize replay buffer with priorities
        if self.prioritized_replay:
            self.replay_buffer = deque(maxlen=replay_capacity)
            self.priorities = deque(maxlen=replay_capacity)
            self.alpha = 0.6  # Priority exponent
            self.beta = 0.4   # Importance sampling weight
            self.beta_increment = 0.001
            self.epsilon_prio = 1e-6
        else:
            self.replay_buffer = deque(maxlen=replay_capacity)
        
        self.update_counter = 0
        
        # Track average TD error for adaptive learning rate
        self.avg_td_error = 0
        self.td_error_decay = 0.99
        
    def add_experience(self, state, action, reward, next_state, done):
        # Scale reward for stability
        scaled_reward = reward * self.reward_scale
        
        # Store experience in n-step buffer
        self.n_step_buffer.append((state, action, scaled_reward, next_state, done))
        
        # Only add to replay buffer if we have enough transitions for n-step return
        if len(self.n_step_buffer) < self.n_step and not done:
            return
            
        # Calculate n-step return
        n_step_reward = 0
        for i in range(len(self.n_step_buffer)):
            n_step_reward += self.gamma**i * self.n_step_buffer[i][2]
        
        # Get state, action from oldest experience and latest next_state, done
        initial_state = self.n_step_buffer[0][0]
        initial_action = self.n_step_buffer[0][1]
        
        # Latest next state is either from the buffer or provided if buffer is full
        if done:
            # If episode terminated, use the latest state as next_state
            final_next_state = next_state
            final_done = True
        else:
            # Use the last state in the buffer
            final_next_state = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]
        
        # Create n-step experience
        experience = (initial_state, initial_action, n_step_reward, final_next_state, final_done)
        
        if self.prioritized_replay:
            # New experiences get max priority
            if len(self.replay_buffer) > 0:
                max_priority = max(self.priorities)
            else:
                max_priority = 1.0
                
            self.replay_buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.replay_buffer.append(experience)
        
        # Clear buffer if episode is done
        if done:
            self.n_step_buffer.clear()
        
    def sample_batch(self):
        if self.prioritized_replay:
            # Convert priorities to probabilities
            probs = np.array(self.priorities) ** self.alpha
            probs /= probs.sum()
            
            # Sample indices according to priorities
            indices = np.random.choice(len(self.replay_buffer), 
                                      min(self.batch_size, len(self.replay_buffer)), 
                                      replace=False, 
                                      p=probs[:len(self.replay_buffer)])
            
            # Get experiences
            batch = [self.replay_buffer[i] for i in indices]
            
            # Calculate importance sampling weights
            weights = (len(self.replay_buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            
            # Increase beta toward 1
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            return batch, indices, weights
        else:
            if len(self.replay_buffer) < self.batch_size:
                batch = random.sample(self.replay_buffer, len(self.replay_buffer))
            else:
                batch = random.sample(self.replay_buffer, self.batch_size)
            return batch, None, None
    
    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon_prio) ** self.alpha
            self.priorities[i] = priority
        
    def soft_update(self):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_net.trainable_variables, 
                                      self.q_net.trainable_variables):
            target_param.assign(target_param * (1.0 - self.tau) + param * self.tau)
    
    def huber_loss_fn(self, y_true, y_pred, delta=1.0):
        """Huber loss - quadratic for small errors, linear for large ones"""
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic**2 + delta * linear
    
    def adjust_learning_rate(self, td_error):
        """Adjusts learning rate based on TD error magnitude"""
        # Update average TD error
        self.avg_td_error = self.td_error_decay * self.avg_td_error + \
                           (1 - self.td_error_decay) * np.mean(np.abs(td_error))
        
        # If TD error is too large, reduce learning rate
        if self.avg_td_error > 5.0:
            # Reduce learning rate
            current_lr = self.optimizer.learning_rate.numpy()
            new_lr = max(current_lr * 0.9, 1e-6)  # Avoid too small learning rate
            self.optimizer.learning_rate.assign(new_lr)
            return True
        return False
        
    def train_step(self):
        if len(self.replay_buffer) < max(self.batch_size, self.n_step):
            return 0
            
        self.update_counter += 1
        if self.update_counter % self.update_freq != 0:
            return 0
        
        # Sample batch with priorities if enabled
        if self.prioritized_replay:
            batch, indices, is_weights = self.sample_batch()
            is_weights = tf.convert_to_tensor(is_weights, dtype=tf.float32)
        else:
            batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
            indices, is_weights = None, None
            
        # Extract batch data
        grids = np.array([b[0]['grid'] for b in batch])
        products = np.array([b[0]['product'] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_grids = np.array([b[3]['grid'] for b in batch])
        next_products = np.array([b[3]['product'] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=bool)
        
        # Double DQN with additional stability measures
        next_q_values = self.q_net((next_grids, next_products), training=False)
        
        # Apply valid action masking for next state
        next_valid_masks = np.array([next_grid.reshape(-1) == 0 for next_grid in next_grids])
        masked_next_q = next_q_values.numpy().copy()
        masked_next_q[~next_valid_masks] = -np.inf
        best_actions = np.argmax(masked_next_q, axis=1)
        
        # Get Q-values from target network
        target_q_values = self.target_net((next_grids, next_products), training=False)
        
        # Gather Q-values for best actions
        batch_indices = np.arange(len(batch))
        target_q = target_q_values.numpy()[batch_indices, best_actions]
        
        # Calculate gamma to the power of n for n-step returns
        n_gamma = self.gamma ** self.n_step
        
        # Compute targets with n-step returns
        targets = rewards + n_gamma * target_q * (~dones)
        
        
        # Compute loss and update online network
        with tf.GradientTape() as tape:
            q_values = self.q_net((grids, products), training=True)
            pred_q = tf.gather_nd(q_values, 
                                tf.stack([tf.range(len(batch), dtype=tf.int64), 
                                        tf.cast(actions, tf.int64)], axis=1))
            
            # Calculate TD errors for prioritized replay update
            td_errors = targets - pred_q.numpy()
            
            # Use Huber loss for robustness
            if self.huber_loss:
                elementwise_loss = self.huber_loss_fn(targets, pred_q, delta=1.0)
            else:
                elementwise_loss = tf.square(td_errors)
                
            # Apply importance sampling weights if using prioritized replay
            if self.prioritized_replay and is_weights is not None:
                loss = tf.reduce_mean(is_weights * elementwise_loss)
            else:
                loss = tf.reduce_mean(elementwise_loss)
        
        # Check if learning rate needs adjustment
        self.adjust_learning_rate(td_errors)
        
        # Update priorities if using prioritized replay
        if self.prioritized_replay and indices is not None:
            self.update_priorities(indices, td_errors)
        
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        
        # Soft update target network
        self.soft_update()
        
        return float(loss)