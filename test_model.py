import os
import tensorflow as tf
import numpy as np
import pandas as pd
from test2 import AnaquelEnv, QNetwork

def test_model():
    # Load the environment
    file_path = 'productos_anaquel.xls'
    df_list = []
    i = 1

    while True:
        try:
            df_list.append(pd.read_excel(file_path, sheet_name=f"Sheet {i}"))
            i += 1
        except Exception:
            break

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all[df_all['ANAQUEL'].str.startswith('C', na=False)]
    df_all = df_all[df_all['CAMPA'] == 201416]
    df_all.reset_index(drop=True, inplace=True)

    env = AnaquelEnv(df_all)
    input_dim = env.state_space
    output_dim = env.action_space

    # Load the model
    q_network = QNetwork(input_dim, output_dim)
    checkpoint_path = "./experiments/checkpoints/model.weights.h5"

    if os.path.exists(checkpoint_path):
        print("Loading saved weights...")
        dummy_input = tf.random.uniform((1, input_dim))
        q_network(dummy_input)  # Build the model
        q_network.load_weights(checkpoint_path)
    else:
        print("No saved model found! Please train the model first.")
        return

    # Test the model
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        q_values = q_network(tf.expand_dims(state, axis=0))[0].numpy()
        action = np.argmax(q_values)
        print(f"Action taken: {action}")
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Total reward achieved during testing: {total_reward}")

if __name__ == "__main__":
    test_model()