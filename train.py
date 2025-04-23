import numpy as np
import tensorflow as tf
from collections import deque
import random
from cnn_dqn_model import CNNPlacementQNetwork
from product_placement_env import ProductPlacementEnv

# === Hyperparámetros ===
EPISODES = 300
MAX_STEPS = 98
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_CAPACITY = 10000
LR = 0.0005
TAU = 0.01

# === Agrupación balanceada de productos ===
def agrupar_productos(df, num_grupos=9):
    grupos = [[] for _ in range(num_grupos)]
    suma_cantidades = [0] * num_grupos
    conteo_productos = [0] * num_grupos
    df = df.sort_values('cantidad', ascending=False).reset_index(drop=True)
    for _, row in df.iterrows():
        mejor_grupo = min(range(num_grupos), key=lambda g: (suma_cantidades[g], conteo_productos[g]))
        grupos[mejor_grupo].append(row)
        suma_cantidades[mejor_grupo] += row['cantidad']
        conteo_productos[mejor_grupo] += 1
    return [pd.DataFrame(g) for g in grupos]

import pandas as pd
np.random.seed(42)
df = pd.DataFrame({
    'cantidad': np.random.randint(1, 100, size=100),
    'alto': np.random.uniform(5, 20, size=100),
    'ancho': np.random.uniform(5, 20, size=100),
    'largo': np.random.uniform(5, 20, size=100)
})

grupos_df = agrupar_productos(df)
productos = grupos_df[0].to_dict('records')  # Usamos solo el grupo 0
env = ProductPlacementEnv(productos)

q_net = CNNPlacementQNetwork()
target_net = CNNPlacementQNetwork()
q_net.build([(None, 14, 7, 1), (None, 5)])
target_net.build([(None, 14, 7, 1), (None, 5)])
target_net.set_weights(q_net.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
replay_buffer = deque(maxlen=REPLAY_CAPACITY)

# === Funciones auxiliares ===
def soft_update(source, target, tau=TAU):
    for t, s in zip(target.trainable_variables, source.trainable_variables):
        t.assign(t * (1.0 - tau) + s * tau)

def get_action(state, epsilon):
    if np.random.rand() < epsilon:
        valid_actions = np.where(state['grid'].reshape(-1) == 0)[0]
        return np.random.choice(valid_actions)
    else:
        grid = tf.convert_to_tensor(state['grid'][None, ...], dtype=tf.float32)
        product = tf.convert_to_tensor(state['product'][None, ...], dtype=tf.float32)
        q_values = q_net((grid, product), training=False)[0].numpy()
        q_values[state['grid'].reshape(-1) == 1] = -1e9  # Mask occupied
        return int(np.argmax(q_values))

# === Entrenamiento principal ===
epsilon = EPSILON_START
for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0
    for step in range(MAX_STEPS):
        action = get_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            print("□ Resultado del episodio:")
            env.render()
            break

        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            grids = np.array([b[0]['grid'] for b in batch], dtype=np.float32)
            products = np.array([b[0]['product'] for b in batch], dtype=np.float32)
            actions = np.array([b[1] for b in batch])
            rewards = np.array([b[2] for b in batch], dtype=np.float32)
            next_grids = np.array([b[3]['grid'] for b in batch], dtype=np.float32)
            next_products = np.array([b[3]['product'] for b in batch], dtype=np.float32)
            dones = np.array([b[4] for b in batch], dtype=bool)

            next_q = q_net((next_grids, next_products), training=False)
            next_actions = tf.argmax(next_q, axis=1)
            target_q = target_net((next_grids, next_products), training=False)
            target_q_values = tf.gather(target_q, next_actions[:, None], batch_dims=1).numpy().flatten()
            targets = rewards + GAMMA * target_q_values * (~dones)

            with tf.GradientTape() as tape:
                q_vals = q_net((grids, products), training=True)
                pred_q = tf.gather(q_vals, actions[:, None], batch_dims=1).numpy().flatten()
                loss = tf.reduce_mean(tf.square(targets - pred_q))

            grads = tape.gradient(loss, q_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
            soft_update(q_net, target_net)

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Ep {ep} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
