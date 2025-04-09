import os
import dotenv
dotenv.load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
import numpy as np
import tensorflow as tf
from tqdm import trange
import pandas as pd
from Assignment import AssignmentEnv
from Dueling_CNN import AssignmentCNNModel
from ReplayBuffer import ReplayBuffer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')


def epsilon_greedy_action(q_values, epsilon, total_cells):
    if np.random.rand() < epsilon:
        return np.random.randint(total_cells, size=q_values.shape[0])
    else:
        return np.argmax(q_values, axis=-1)

def epsilon_greedy_action_soft(q_values, epsilon, total_cells):
    flg_use_epsilon = False
    if np.random.rand() < epsilon:
        return np.random.randint(total_cells, size=q_values.shape[0]), flg_use_epsilon
    else:
        # Apply softmax to each product's q-values and sample
        probabilities = tf.nn.softmax(q_values, axis=-1).numpy()
        actions = [np.random.choice(total_cells, p=prob) for prob in probabilities]
        return np.array(actions), flg_use_epsilon


def build_index_batch(actions):
    batch_indices = tf.range(tf.shape(actions)[0])
    return tf.stack([batch_indices, actions], axis=1)


def train_double_dqn(env: AssignmentEnv, q_network, target_network, num_episodes=1000, gamma=0.99,
                     lr=1e-4, batch_size=64, buffer_capacity=10000,
                     epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.98,
                     target_update_freq=10):

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)
    rewards_list = []

    epsilon = epsilon_start
    tr = trange(num_episodes)
    for episode in tr:
        product_features, matrix_input, product_mask = env.reset()

        state = (
            product_features.astype(np.float32),
            matrix_input.astype(np.float32),
            product_mask.astype(np.float32)
        )

        batch_input = (
            np.expand_dims(state[0], 0),
            np.expand_dims(state[1], 0),
            np.expand_dims(state[2], 0)
        )

        _, q_values = q_network(batch_input)
        q_values = q_values.numpy()[0]
        action_vector, flg_use_epsilon = epsilon_greedy_action_soft(q_values, epsilon, env.total_cells)

        grid, reward, done, _, placed_estimated_sales = env.step(action_vector)
        items_placed = np.sum(grid != -1)

        next_state = env.reset()
        next_input = (
            next_state[0].astype(np.float32),
            next_state[1].astype(np.float32),
            next_state[2].astype(np.float32)
        )

        replay_buffer.push(state, action_vector, reward, next_input, done)

        loss_value = None
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = batch

            product_feats_batch = np.array([s[0] for s in states], dtype=np.float32)
            matrix_input_batch = np.array([s[1] for s in states], dtype=np.float32)
            mask_batch = np.array([s[2] for s in states], dtype=np.float32)

            next_product_feats_batch = np.array([s[0] for s in next_states], dtype=np.float32)
            next_matrix_input_batch = np.array([s[1] for s in next_states], dtype=np.float32)
            next_mask_batch = np.array([s[2] for s in next_states], dtype=np.float32)

            actions_batch = np.array(actions)
            rewards_batch = np.array(rewards, dtype=np.float32)
            dones_batch = np.array(dones, dtype=np.float32)

            rewards_batch = tf.expand_dims(rewards_batch, axis=-1)
            dones_batch = tf.expand_dims(dones_batch, axis=-1)

            states_input = (product_feats_batch, matrix_input_batch, mask_batch)
            next_state_input = (next_product_feats_batch, next_matrix_input_batch, next_mask_batch)

            with tf.GradientTape() as tape:
                _, q_values = q_network(states_input)
                _, next_q_values = q_network(next_state_input)
                next_actions = tf.argmax(next_q_values, axis=-1)

                _, target_q_values = target_network(next_state_input)
                num_products = tf.shape(next_actions)[1].numpy()
                batch_size = tf.shape(next_actions)[0].numpy()

                batch_idx = tf.range(batch_size, dtype=tf.int32)
                product_idx = tf.range(num_products, dtype=tf.int32)

                batch_idx = tf.reshape(batch_idx, (-1, 1))
                batch_idx = tf.tile(batch_idx, [1, num_products])

                product_idx = tf.reshape(product_idx, (1, -1))
                product_idx = tf.tile(product_idx, [batch_size, 1])

                cell_idx = tf.cast(next_actions, tf.int32)
                indices = tf.stack([batch_idx, product_idx, cell_idx], axis=-1)

                target_q = tf.gather_nd(target_q_values, indices)
                target_q = tf.clip_by_value(target_q, -10.0, 10.0)

                actions_one_hot = tf.one_hot(actions_batch, env.total_cells)
                chosen_q = tf.reduce_sum(q_values * actions_one_hot, axis=-1)

                td_target = rewards_batch + (1.0 - dones_batch) * gamma * target_q
                loss = tf.reduce_mean(tf.keras.losses.huber(td_target, chosen_q))
                loss_value = loss.numpy()

            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

        if episode % target_update_freq == 0:
            target_network.set_weights(q_network.get_weights())

        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        tr.set_description(
            f"Ep {episode+1} | epsilon: {epsilon:2f} | use epsilon: {flg_use_epsilon} | Reward: {reward:.2f} | Placed: {items_placed} | Sales: {placed_estimated_sales:.1f} | Loss: {loss_value:.4f}" 
            if loss_value is not None else f"Ep {episode+1}"
        )
        rewards_list.append([reward, items_placed, placed_estimated_sales, loss_value, flg_use_epsilon])

    return rewards_list

import itertools
from tqdm import tqdm

alpha_vals = [10.0, 50.0, 100.0]
beta_vals = [0.5, 1.0]
gamma_vals = [0.5, 1.0]
delta_vals = [0.5, 1.0]
theta_vals = [0.2, 0.7, 2.0]

if __name__ == "__main__":
    file_path = 'productos_anaquel.xlsx'
    df_list = []
    i = 1
    try:
        while True:
            df_list.append(pd.read_excel(file_path, sheet_name=f"Sheet {i}"))
            i += 1
    except Exception:
        pass

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all[df_all['ANAQUEL'].str.startswith('C', na=False)]
    df_by_campa = [df_all[df_all['CAMPA'] == campa].copy() for campa in df_all['CAMPA'].unique()]


    for i, df in enumerate(df_by_campa):
        df.loc[df['UNDESTIMADAS'] <= 0, 'UNDESTIMADAS'] = 1
        df.reset_index(drop=True, inplace=True)
        df.loc[:, 'PRODUCTO_NORM'] = (df['PRODUCTO'] - df['PRODUCTO'].min()) / (df['PRODUCTO'].max() - df['PRODUCTO'].min())
        df.loc[:, 'UNDESTIMADAS_NORM'] = (df['UNDESTIMADAS'] - df['UNDESTIMADAS'].min()) / (df['UNDESTIMADAS'].max() - df['UNDESTIMADAS'].min())
        df = df[['PRODUCTO_NORM', 'UNDESTIMADAS_NORM']]
        df_by_campa[i] = df

    for alpha, beta, gamma, delta, theta in tqdm(list(itertools.product(alpha_vals, beta_vals, gamma_vals, delta_vals, theta_vals))):
        env = AssignmentEnv(df_by_campa, alpha=alpha, beta=beta, gamma=gamma, delta=delta, theta=theta)
        q_net = AssignmentCNNModel(env.rows, env.cols, product_feature_dim=2, matrix_channels=1, embed_dim=256)
        target_net = AssignmentCNNModel(env.rows, env.cols, product_feature_dim=2, matrix_channels=1, embed_dim=256)

        dummy_product_features = tf.random.uniform((1, env.total_cells, 2))
        dummy_matrix_input = tf.random.uniform((1, env.rows, env.cols, 1))
        dummy_product_mask = tf.ones((1, env.total_cells), dtype=tf.float32)

        dummy_input = (dummy_product_features, dummy_matrix_input, dummy_product_mask)
        _ = q_net(dummy_input)
        _ = target_net(dummy_input)

        target_net.set_weights(q_net.get_weights())

        rewards = train_double_dqn(env, q_net, target_net, num_episodes=200)

        import matplotlib.pyplot as plt

        # Plot rewards
        plt.plot([r[0] for r in rewards])
        plt.title(f"Reward per episode - Double DQN\nAlpha: {alpha}, Beta: {beta}, Gamma: {gamma}, Delta: {delta}, Theta: {theta}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid()
        plt.savefig(f"./experiments/reward/DoubleDQN_reward_plot_alpha_{alpha}_beta_{beta}_gamma_{gamma}_delta_{delta}_theta_{theta}.png")
        plt.close()

        # Plot items placed
        plt.plot([r[1] for r in rewards])
        plt.title(f"Items Placed per episode - Double DQN\nAlpha: {alpha}, Beta: {beta}, Gamma: {gamma}, Delta: {delta}, Theta: {theta}")
        plt.xlabel("Episode")
        plt.ylabel("Items Placed")
        plt.grid()
        plt.savefig(f"./experiments/items_placed/DoubleDQN_items_placed_plot_alpha_{alpha}_beta_{beta}_gamma_{gamma}_delta_{delta}_theta_{theta}.png")
        plt.close()

        # Plot placed estimated sales
        plt.plot([r[2] for r in rewards])
        plt.title(f"Placed Estimated Sales per episode - Double DQN\nAlpha: {alpha}, Beta: {beta}, Gamma: {gamma}, Delta: {delta}, Theta: {theta}")
        plt.xlabel("Episode")
        plt.ylabel("Placed Estimated Sales")
        plt.grid()
        plt.savefig(f"./experiments/sales/DoubleDQN_placed_sales_plot_alpha_{alpha}_beta_{beta}_gamma_{gamma}_delta_{delta}_theta{theta}.png")
        plt.close()

        # Plot loss value
        plt.plot([r[3] for r in rewards if r[3] is not None])
        plt.title(f"Loss per episode - Double DQN\nAlpha: {alpha}, Beta: {beta}, Gamma: {gamma}, Delta: {delta}, Theta: {theta}")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig(f"./experiments/loss/DoubleDQN_loss_plot_alpha_{alpha}_beta_{beta}_gamma_{gamma}_delta_{delta}_theta_{theta}.png")
        plt.close()

        # Plot epsilon usage
        plt.plot([r[4] for r in rewards])
        plt.title(f"Epsilon Usage per episode - Double DQN\nAlpha: {alpha}, Beta: {beta}, Gamma: {gamma}, Delta: {delta}, Theta: {theta}")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Usage")
        plt.grid()
        plt.savefig(f"./experiments/epsilon_usage/DoubleDQN_epsilon_usage_plot_alpha_{alpha}_beta_{beta}_gamma_{gamma}_delta_{delta}_theta_{theta}.png")
        plt.close()
