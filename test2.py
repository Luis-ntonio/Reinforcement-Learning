import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import random
from collections import deque, namedtuple
import os
import logging
import pandas as pd

class AnaquelEnv:
    def __init__(self, df, rows=3, cols=7):
        self.df = df.copy()
        # For simplicity, we assume the weight matrix dimensions match our grid dimensions.
        self.rows = self.weight_matrix.shape[0]
        self.cols = self.weight_matrix.shape[1]
        # Matrices to track placements
        self.avail_matrix = np.zeros(self.weight_matrix.shape)  # 0: free, 1: filled
        self.products_id = np.full(self.weight_matrix.shape, -1)  # -1 means no product

        # Mapping product IDs to indexes for one-hot encoding
        unique_products = df['PRODUCTO'].unique()
        self.product_id_to_index = {pid: idx for idx, pid in enumerate(unique_products)}
        self.num_products = len(unique_products)

        # The state will be represented as a flattened vector of shape:
        # (rows * cols) * (1 + num_products)
        self.state_space = self.rows * self.cols * (1 + self.num_products)
        # The action space: choose a product (from the df rows) and choose a cell (from rows*cols)
        self.action_space = (self.rows * self.cols) * self.num_products

        self.failed_attempts = 0
        self.state_quantities = np.zeros(self.weight_matrix.shape)

        # Calculate the maximum possible total cost for normalization
        self.max_possible_total_cost = self.calculate_max_possible_cost()
        self.max_possible_placement_reward = np.log(np.max(self.df['UNDESTIMADAS']) * np.max(self.weight_matrix))

    def reset(self):
        """Reset environment for a new episode."""
        self.state_quantities = np.zeros(self.weight_matrix.shape)
        self.avail_matrix.fill(0)
        self.products_id.fill(-1)
        self.failed_attempts = 0
        return self.get_state()

    def get_state(self):
        """
        Returns a flattened state.
        For each cell, the first element is the quantity (or 0 if empty),
        and the remaining are a one-hot encoding of the product placed (all zeros if none).
        """
        state = np.zeros((self.rows, self.cols, 1 + self.num_products), dtype=np.float32)
        # First channel: quantity
        state[:, :, 0] = self.state_quantities
        # One-hot channels for product
        for i in range(self.rows):
            for j in range(self.cols):
                pid = self.products_id[i, j]
                if pid != -1:
                    idx = self.product_id_to_index.get(pid, None)
                    if idx is not None:
                        state[i, j, idx + 1] = 1.0
        return state.flatten()

    def step(self, action):
        """
        Maps the action integer into a product selection and a fixed cell coordinate.
        If the chosen cell is empty, the product is placed.
        If the cell is already occupied, a penalty is applied.
        Returns next_state, reward, done.
        """
        total_cells = self.rows * self.cols
        item = action // total_cells
        cell = action % total_cells
        row, col = divmod(cell, self.cols)

        # Get product information from dataframe
        product_id = self.df.iloc[item]['PRODUCTO']
        quantity = self.df.iloc[item]['UNDESTIMADAS']

        if self.avail_matrix[row, col] == 0:
            # Check if product_id is already placed
            if product_id in self.products_id:
                self.failed_attempts += 1
                base_penalty = self.max_possible_placement_reward * 10
                # Increasing penalty for repeated attempts to encourage exploration
                reward = -base_penalty * (1 + 0.1 * self.failed_attempts)
                
                if self.failed_attempts > 10:
                    done = True
                    # Even larger terminal penalty
                    reward = -base_penalty * 20
                    return self.get_state(), reward, done
            else:
                self.failed_attempts = 0
                self.products_id[row, col] = product_id
                self.state_quantities[row, col] = quantity
                self.avail_matrix[row, col] = 1
                reward = -np.log( quantity * self.weight_matrix[row, col])   
        else:
            reward = -self.max_possible_placement_reward * 8  # Large penalty for occupied cell

        done = self.is_done()
        if done:
            print("All products placed. Episode complete.")
            reward += 100 - (np.sum(self.state_quantities * self.weight_matrix)/self.max_possible_total_cost)
        next_state = self.get_state()
        return next_state, reward, done

    def calculate_max_possible_cost(self):
        """
        Calculate the maximum possible total cost if all products were placed
        in the worst possible positions.
        """
        # Sort products by quantity in descending order
        sorted_products = self.df.sort_values(by='UNDESTIMADAS', ascending=False)
        
        # Sort cells by weight in descending order (higher weight = worse position)
        flat_weights = self.weight_matrix.flatten()
        sorted_cell_indices = np.argsort(flat_weights)[::-1]  # Reverse to get descending order
        
        max_cost = 0
        products_placed = 0
        
        # Place each product in the worst possible cell (highest weight)
        for i, product in sorted_products.iterrows():
            if products_placed >= len(sorted_cell_indices):
                break
                
            quantity = product['UNDESTIMADAS']
            cell_index = sorted_cell_indices[products_placed]
            row, col = np.unravel_index(cell_index, self.weight_matrix.shape)
            
            # Calculate cost for this worst-case placement
            max_cost += quantity * self.weight_matrix[row, col]
            products_placed += 1
            
            # Stop if we've placed all products or run out of cells
            if products_placed >= min(len(sorted_products), self.rows * self.cols):
                break
        
        return max_cost

    def is_done(self):
        """Episode is done when all cells are filled (or if a certain number of products are placed)."""
        # NOTE: Ensure that the termination condition aligns with your problem setup.
        return np.sum(self.avail_matrix) == self.num_products





class QNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.fc4 = tf.keras.layers.Dense(output_dim, activation=None)
    
    def build(self, input_shape):
        # Explicitly build all layers, even if not used later
        self.fc1.build(input_shape)
        # If you plan to use fc2 later, build it too
        self.fc2.build(self.fc1.compute_output_shape(input_shape))
        fc1_out_shape = self.fc1.compute_output_shape(input_shape)
        self.fc3.build(fc1_out_shape)
        fc3_out_shape = self.fc3.compute_output_shape(fc1_out_shape)
        self.fc4.build(fc3_out_shape)
        super(QNetwork, self).build(input_shape)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)


def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator(tf.expand_dims(observation, axis=0))[0].numpy()
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def update_target_network(q_network, target_q_network):
    target_q_network.set_weights(q_network.get_weights())


def deep_q_learning(env: AnaquelEnv,
                    q_estimator: QNetwork,
                    target_estimator: QNetwork,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=5000,
                    replay_memory_init_size=1000,
                    update_target_estimator_every=50,
                    discount_factor=0.99,
                    epsilon_start=0.80,
                    epsilon_end=0.01,
                    epsilon_decay_steps=10000,
                    batch_size=64):
    
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = deque(maxlen=replay_memory_size)
    rewards_list = []

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model.weights.h5")

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(q_estimator, env.action_space)

    print("Populating replay memory...")
    state = env.reset()
    for i in range(replay_memory_init_size):
        epsilon = epsilons[min(i, epsilon_decay_steps-1)]
        action_probs = policy(state, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done = env.step(action)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        state = env.reset() if done else next_state
    print("Replay memory initialized.")

    # Use gradient clipping in the optimizer to prevent exploding gradients.
    optimizer = Adam(learning_rate=0.0005, clipnorm=0.001)

    global_step = 0  # initialize global step
    # Training loop
    for episode in range(num_episodes):
        local_step = 0
        state = env.reset()
        total_reward = 0
        done = False

        # Generation trainning: each loop is one item picked from the list. it will be done until all items are placed
        while not done:
            epsilon = epsilons[global_step%epsilon_decay_steps]
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.step(action)
            replay_memory.append(Transition(state, action, reward, next_state, done))

            if len(replay_memory) >= batch_size:
                batch = random.sample(replay_memory, batch_size)  
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
                dones = tf.convert_to_tensor(np.array(dones, dtype=np.float32), dtype=tf.float32)

                with tf.GradientTape() as tape:
                    q_values = q_estimator(states)
                    q_values = tf.gather(q_values, actions, batch_dims=1)

                    online_next_q_values = q_estimator(next_states)
                    best_next_actions = tf.argmax(online_next_q_values, axis=1)
                    
                    target_next_q_values = target_estimator(next_states)
                    double_q_values = tf.gather(target_next_q_values, best_next_actions, batch_dims=1)
                    
                    targets = rewards + discount_factor * double_q_values * (1 - dones)
                    targets = tf.clip_by_value(targets, -10.0, 10.0)
                    loss = tf.reduce_mean(tf.keras.losses.Huber()(targets, q_values))
                grads = tape.gradient(loss, q_estimator.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_estimator.trainable_variables))
            
            state = next_state
            total_reward += reward
            if local_step > env.num_products * 2:
                break
            local_step += 1
            global_step += 1
            print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {epsilon:.4f}, Loss: {loss:.4f} Items placed: {np.sum(env.avail_matrix)}")


        rewards_list.append(total_reward)

        if episode % update_target_estimator_every == 0:
            update_target_network(q_estimator, target_estimator)

        q_estimator.save_weights(checkpoint_path)
        logger.info(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {epsilon:.4f}, Loss: {loss:.4f} Items placed: {np.sum(env.avail_matrix)}")

    return rewards_list




if __name__ == "__main__":
        # Logging and data loading
    file_path = 'productos_anaquel.xls'
    df_list = []
    i = 1

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('log.txt')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('Reading file...')
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
    print(df_all[df_all['UNDESTIMADAS'] == 0])
    df_all.reset_index(drop=True, inplace=True)
    
    experiment_dir = "./experiments"
    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', "model.weights.h5")

    env = AnaquelEnv(df_all)
    num_products = env.num_products
    input_dim = env.state_space  # (rows*cols)*(1+num_products)
    output_dim = env.action_space  # (rows*cols)*num_products

    q_network = QNetwork(input_dim, output_dim)
    target_q_network = QNetwork(input_dim, output_dim)
    target_q_network.set_weights(q_network.get_weights())

    if os.path.exists(checkpoint_path):
        print("Loading saved weights...")
        dummy_input = tf.random.uniform((1, input_dim))
        q_network(dummy_input)
        target_q_network(dummy_input)
        q_network.load_weights(checkpoint_path)
        target_q_network.load_weights(checkpoint_path)
    else:
        print("No saved model found! Train the model first.")
    
    rewards_list = deep_q_learning(env, q_network, target_q_network, num_episodes=1000, experiment_dir=experiment_dir)
