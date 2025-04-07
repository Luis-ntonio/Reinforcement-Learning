import numpy as np
import tensorflow as tf
from tqdm import trange
import os
import pandas as pd
from Assignment import AssignmentEnv
from Dueling_CNN import AssignmentCNNModel
from ReplayBuffer import ReplayBuffer


def epsilon_greedy_action(q_values, epsilon, total_cells):
    if np.random.rand() < epsilon:
        return np.random.randint(total_cells, size=q_values.shape[0])
    else:
        return np.argmax(q_values, axis=-1)


def build_index_batch(actions):
    batch_indices = tf.range(tf.shape(actions)[0])
    return tf.stack([batch_indices, actions], axis=1)


def train_double_dqn(env, q_network, target_network, num_episodes=1000, gamma=0.99,
                     lr=1e-4, batch_size=64, buffer_capacity=10000,
                     epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
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
        action_vector = epsilon_greedy_action(q_values, epsilon, env.total_cells)

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
                num_products = tf.shape(next_actions)[1]

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
        rewards_list.append(reward)

        tr.set_description(
            f"Ep {episode+1} | Reward: {reward:.2f} | Placed: {items_placed} | Sales: {placed_estimated_sales:.1f} | Loss: {loss_value:.4f}" 
            if loss_value is not None else f"Ep {episode+1}"
        )

    return rewards_list

import itertools
from tqdm import tqdm

alpha_vals = [10.0, 50.0, 100.0]
beta_vals = [0.5, 1.0]
gamma_vals = [0.5, 1.0]
delta_vals = [0.5, 1.0]

if __name__ == "__main__":
    file_path = 'productos_anaquel.xls'
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
    df_by_campa = [df_all[df_all['CAMPA'] == campa] for campa in df_all['CAMPA'].unique()]

    for i, df in enumerate(df_by_campa):
        df['UNDESTIMADAS'] = df['UNDESTIMADAS'].apply(lambda x: x if x > 0 else 1)
        df.reset_index(drop=True, inplace=True)
        df['PRODUCTO_NORM'] = (df['PRODUCTO'] - df['PRODUCTO'].min()) / (df['PRODUCTO'].max() - df['PRODUCTO'].min())
        df['UNDESTIMADAS_NORM'] = (df['UNDESTIMADAS'] - df['UNDESTIMADAS'].min()) / (df['UNDESTIMADAS'].max() - df['UNDESTIMADAS'].min())
        df = df[['PRODUCTO_NORM', 'UNDESTIMADAS_NORM']]
        df_by_campa[i] = df

    for alpha, beta, gamma, delta in tqdm(list(itertools.product(alpha_vals, beta_vals, gamma_vals, delta_vals))):
        env = AssignmentEnv(df_by_campa, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        q_net = AssignmentCNNModel(env.rows, env.cols, product_feature_dim=2, matrix_channels=1, embed_dim=256)
        target_net = AssignmentCNNModel(env.rows, env.cols, product_feature_dim=2, matrix_channels=1, embed_dim=256)

        dummy_product_features = tf.random.uniform((1, env.total_cells, 2))
        dummy_matrix_input = tf.random.uniform((1, env.rows, env.cols, 1))
        dummy_product_mask = tf.ones((1, env.total_cells), dtype=tf.float32)

        dummy_input = (dummy_product_features, dummy_matrix_input, dummy_product_mask)
        _ = q_net(dummy_input)
        _ = target_net(dummy_input)

        target_net.set_weights(q_net.get_weights())

        rewards = train_double_dqn(env, q_net, target_net, num_episodes=1000)

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title("Reward per episode - Double DQN")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig("./experiments/DoubleDQN_reward_plot.png")
