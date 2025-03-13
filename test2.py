import numpy as np
import tensorflow as tf
import random
from collections import deque, namedtuple
import os
import logging
import pandas as pd

class AnaquelEnv:
    def __init__(self, df, rows=3, cols=7):
        self.df = df.copy()
        self.df_iterations = df.copy()
        self.weight_matrix = np.array([
            [5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5],
            [5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0],
            [4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3],
            [9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0],
            [9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8],
            [10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5],
        ])
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

    def reset(self):
        """Reset environment for a new episode."""
        self.df_iterations = self.df.copy()
        self.state_quantities = np.zeros(self.weight_matrix.shape)
        self.avail_matrix.fill(0)
        self.products_id.fill(-1)
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
        # Map action to product (item) and cell.
        total_cells = self.rows * self.cols
        item = action // total_cells
        cell = action % total_cells
        row, col = divmod(cell, self.cols)

        # Get product information from dataframe
        product_id = self.df_iterations.iloc[item]['PRODUCTO']
        quantity = self.df_iterations.iloc[item]['UNDESTIMADAS']

        # Check if the cell is available
        if self.avail_matrix[row, col] == 0:
            # Place the product
            self.products_id[row, col] = product_id
            self.state_quantities[row, col] = quantity
            self.avail_matrix[row, col] = 1
            reward = - quantity * self.weight_matrix[row, col]
        else:
            # Penalty for trying to place in an occupied cell
            reward = -5000

        done = self.is_done()
        next_state = self.get_state()
        return next_state, reward, done

    def compute_reward(self, row, col):
        """
        Compute reward based on the quantity and weight for the given cell.
        (This helper is used in the step function above.)
        """
        return - self.state_quantities[row, col] * self.weight_matrix[row, col]

    def is_done(self):
        """Episode is done when all cells are filled."""
        return np.all(self.avail_matrix == 1)


# Set up logging and load your data (adjust file path and conditions as needed)
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
df_all.reset_index(drop=True, inplace=True)


class QNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.fc4 = tf.keras.layers.Dense(output_dim, activation=None)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on the Q-network.
    """
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
                    update_target_estimator_every=500,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=50000,
                    batch_size=32):
    
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = deque(maxlen=replay_memory_size)
    rewards_list = []

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model.weights.h5")

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(q_estimator, env.action_space)

    # Populate replay memory with initial random experience.
    print("Populating replay memory...")
    state = env.reset()
    for i in range(replay_memory_init_size):
        epsilon = epsilons[min(i, epsilon_decay_steps-1)]
        action_probs = policy(state, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        next_state, reward, done = env.step(action)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
        else:
            state = next_state
    print("Replay memory initialized.")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            epsilon = epsilons[min(step_count, epsilon_decay_steps - 1)]
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
                    # Use tf.gather with batch_dims=1 to select the appropriate Q-value for each transition.
                    q_values = tf.gather(q_values, actions, batch_dims=1)
                    next_q_values = target_estimator(next_states)
                    max_next_q_values = tf.reduce_max(next_q_values, axis=1)
                    targets = rewards + discount_factor * max_next_q_values * (1 - dones)
                    loss = tf.keras.losses.MSE(targets, q_values)
                grads = tape.gradient(loss, q_estimator.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_estimator.trainable_variables))
            
            state = next_state
            total_reward += reward
            step_count += 1

        rewards_list.append(total_reward)

        if episode % update_target_estimator_every == 0:
            update_target_network(q_estimator, target_estimator)

        q_estimator.save_weights(checkpoint_path)
        print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    return rewards_list


experiment_dir = "./experiments"
checkpoint_path = os.path.join(experiment_dir, 'checkpoints', "model.weights.h5")

# Initialize environment
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

# Train the agent
rewards_list = deep_q_learning(env, q_network, target_q_network, num_episodes=500, experiment_dir=experiment_dir)
