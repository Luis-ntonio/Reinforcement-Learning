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
from Anaquel import AnaquelEnv
from Dueling_QNet import DuelingQNetwork



gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled on all GPUs.")
    except RuntimeError as e:
        print(e)



def deep_q_learning(env: AnaquelEnv,
                    q_estimator: tf.keras.Model,
                    target_estimator: tf.keras.Model,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=1000,
                    replay_memory_init_size=1000,
                    update_target_estimator_every=50,
                    discount_factor=0.99,
                    temperature_start=1.0,
                    temperature_end=0.1,
                    temperature_decay_steps=10000,
                    batch_size=8,
                    n_step=3):
    # Initialize prioritized replay buffer and n-step buffer
    replay_buffer = PrioritizedReplayBuffer(replay_memory_size)
    n_step_buffer = NStepTransitionBuffer(n_step, discount_factor)
    rewards_list = []

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model.weights.h5")

    global_step = 0

    # Populate replay memory with initial random experience
    print("Populating replay memory...")
    state = env.reset()
    while len(replay_buffer.buffer) < replay_memory_init_size:
        # Temperature decays linearly over steps
        temperature = max(temperature_end, temperature_start - (temperature_start - temperature_end) * global_step % temperature_decay_steps)
        action_probs = softmax_policy(q_estimator, state, temperature)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done = env.step(action)
        transition = Transition(state, action, reward, next_state, done)
        n_step_buffer.add(transition)
        if n_step_buffer.is_ready():
            multi_step_transition = n_step_buffer.get_n_step_transition()
            replay_buffer.add(multi_step_transition)
            n_step_buffer.reset()
        state = env.reset() if done else next_state
        global_step += 1
    print("Replay memory initialized.")

    from tensorflow.keras.mixed_precision import LossScaleOptimizer
    base_optimizer = Adam(learning_rate=0.0005, clipnorm=0.0001)
    optimizer = LossScaleOptimizer(base_optimizer)

    beta_start = 0.4
    beta_frames = 10000

    # Main training loop
    for episode in range(num_episodes):
        local_step = 0
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            temperature = max(temperature_end, temperature_start - (temperature_start - temperature_end) * local_step / temperature_decay_steps)
            action_probs = softmax_policy(q_estimator, state, temperature)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.step(action)
            transition = Transition(state, action, reward, next_state, done)
            n_step_buffer.add(transition)
            if n_step_buffer.is_ready():
                multi_step_transition = n_step_buffer.get_n_step_transition()
                replay_buffer.add(multi_step_transition)
                n_step_buffer.reset()
            if done and n_step_buffer.buffer:
                # Flush remaining transitions at end of episode
                multi_step_transition = n_step_buffer.get_n_step_transition()
                replay_buffer.add(multi_step_transition)
                n_step_buffer.reset()
            state = next_state
            total_reward += reward

            if len(replay_buffer.buffer) >= batch_size:
                beta = min(1.0, beta_start + global_step * (1.0 - beta_start) / beta_frames)
                samples, indices, weights = replay_buffer.sample(batch_size, beta)
                states, actions, rewards, next_states, dones = zip(*samples)

                states = tf.convert_to_tensor(np.array(states), dtype=tf.float16)
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float16)
                next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float16)
                dones = tf.convert_to_tensor(np.array(dones, dtype=np.float16), dtype=tf.float16)
                weights = tf.convert_to_tensor(weights, dtype=tf.float16)

                with tf.GradientTape() as tape:
                    q_values = q_estimator(states)
                    q_values = tf.gather(q_values, actions, batch_dims=1)

                    # Double DQN: select best next actions using the online network
                    online_next_q = q_estimator(next_states)
                    best_next_actions = tf.argmax(online_next_q, axis=1)

                    # Evaluate the Q-value of these actions using the target network
                    target_next_q = target_estimator(next_states)
                    double_q = tf.gather(target_next_q, best_next_actions, batch_dims=1)

                    # Cast rewards and dones to the same dtype as double_q (likely float16)
                    rewards = tf.cast(rewards, dtype=double_q.dtype)
                    dones = tf.cast(dones, dtype=double_q.dtype)
                    discount = tf.constant(discount_factor ** n_step, dtype=double_q.dtype)

                    targets = rewards + discount * double_q * (1 - dones)
                    targets = tf.clip_by_value(targets, -10.0, 10.0)
                    loss = tf.reduce_mean(tf.keras.losses.Huber()(targets, q_values))

                grads = tape.gradient(loss, q_estimator.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_estimator.trainable_variables))
                # Update priorities based on absolute TD error
                td_errors = tf.abs(targets - q_values).numpy().flatten() + 1e-6
                replay_buffer.update_priorities(indices, td_errors)

            total_reward += reward
            local_step += 1
            global_step += 1
            print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Temperature: {temperature:.4f}, Loss: {loss:.4f}, Items placed: {np.sum(env.avail_matrix)}")
            if local_step > env.num_products * 2:
                break

        rewards_list.append(total_reward)
        if episode % update_target_estimator_every == 0:
            update_target_network(q_estimator, target_estimator)
        q_estimator.save_weights(checkpoint_path)
        logger.info(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Temperature: {temperature:.4f}, Loss: {loss:.4f}, Items placed: {np.sum(env.avail_matrix)}")
    return rewards_list

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    # Logging configuration and data loading
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
    input_dim = env.state_space    # (rows*cols)*(1+num_products)
    output_dim = env.action_space   # (rows*cols)*num_products

    # Instantiate dueling Q-networks with noisy layers enabled
    q_network = DuelingQNetwork(input_dim, output_dim, use_noisy=True)
    target_q_network = DuelingQNetwork(input_dim, output_dim, use_noisy=True)
    target_q_network.set_weights(q_network.get_weights())

    if os.path.exists(checkpoint_path):
        print("Loading saved weights...")
        dummy_input = tf.random.uniform((1, input_dim))
        q_network(dummy_input)
        target_q_network(dummy_input)
        q_network.load_weights(checkpoint_path)
        target_q_network.load_weights(checkpoint_path)
    else:
        print("No saved model found! Training from scratch...")

    rewards_list = deep_q_learning(env, q_network, target_q_network,
                                   num_episodes=1000, experiment_dir=experiment_dir)
