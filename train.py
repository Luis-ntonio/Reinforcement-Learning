import pandas as pd
import numpy as np
import tensorflow as tf
from agent import EnhancedDQNAgent  # Use our improved agent
import random
from cnn_dqn_model import create_improved_model, get_data
from product_placement_env import ProductPlacementEnv
import matplotlib.pyplot as plt
import os
from utils import agrupar_productos
from hyperparameters import *

# Create a directory for saving results
os.makedirs("results", exist_ok=True)

def get_action(state, epsilon, q_net, env):
    """Get action with epsilon-greedy policy with improved handling of invalid actions"""
    if np.random.rand() < epsilon:
        # For exploration, only consider valid actions
        valid_actions = np.where(state['grid'].reshape(-1) == 0)[0]
        if len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        return 0  # Fallback, should not happen if environment is working correctly
    else:
        grid = tf.convert_to_tensor(state['grid'][None, ...], dtype=tf.float32)
        product = tf.convert_to_tensor(state['product'][None, ...], dtype=tf.float32)
        q_values = q_net((grid, product), training=False)[0].numpy()
        
        # Create a mask for valid actions (unoccupied cells)
        mask = (state['grid'].reshape(-1) == 0)
        if not np.any(mask):  # No valid actions
            return 0  # Fallback
        
        # Apply mask and get best valid action
        masked_q = np.copy(q_values)
        masked_q[~mask] = -np.inf
        return int(np.argmax(masked_q))

def evaluate_agent(agent, grupos_df, num_eval=3):
    """Evaluate agent performance without exploration"""
    eval_rewards = []
    
    for _ in range(num_eval):
        grupo_idx = np.random.randint(0, len(grupos_df))
        productos = grupos_df[grupo_idx]
        ids = productos[['PRODUCTO']].to_dict('records')
        productos = productos[['ALTO', 'ANCHO', 'LARGO', 'VOLUMEN', 'PESO', "UNDESTIMADAS"]].to_dict('records')
        env = ProductPlacementEnv(productos, ids)
        
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Always use greedy action selection during evaluation
            grid = tf.convert_to_tensor(state['grid'][None, ...], dtype=tf.float32)
            product = tf.convert_to_tensor(state['product'][None, ...], dtype=tf.float32)
            q_values = agent.q_net((grid, product), training=False)[0].numpy()
            
            # Only consider valid actions
            mask = (state['grid'].reshape(-1) == 0)
            masked_q = np.copy(q_values)
            masked_q[~mask] = -np.inf
            action = int(np.argmax(masked_q))
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
        
        # Store raw reward without scaling
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Get and group product data
    df = get_data()
    
    # Use our improved model architecture with batch normalization
    q_net = create_improved_model()
    target_net = create_improved_model()
    
    # Copy weights from online to target network
    target_net.set_weights(q_net.get_weights())
    
    # Use Adam optimizer with lower learning rate and gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR) #, clipnorm=1.0)
    
    # Create improved agent with prioritized experience replay and Huber loss
    agent = EnhancedDQNAgent(
        q_net=q_net,
        target_net=target_net,
        optimizer=optimizer,
        gamma=GAMMA,
        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
        update_freq=UPDATE_FREQ,
        tau=TAU,
        reward_scale=REWARD_SCALE,
        grad_clip=0.5,
        prioritized_replay=True,
        huber_loss=True,
        n_step_returns=3
    )
    
    # Training metrics
    rewards_history = []
    eval_rewards_history = []
    loss_history = []
    epsilon = EPSILON_START
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    
    campas = df['CAMPA'].unique()
    # === Training loop ===
    for campa in campas:
        print(f"Training for campaign: {campa}")
        df_campa = df[df['CAMPA'] == campa].reset_index(drop=True)
        Anaqueles = df_campa[['ANAQUEL']].to_dict('records')
        grupos_df = agrupar_productos(df_campa)
        for ep in range(EPISODES):
            grupo_idx = np.random.randint(0, len(grupos_df))
            productos = grupos_df[grupo_idx]
            ids = productos[['PRODUCTO']].to_dict('records')
            productos = productos[['ALTO', 'ANCHO', 'LARGO', 'VOLUMEN', 'PESO', "UNDESTIMADAS"]].to_dict('records')
            env = ProductPlacementEnv(productos, ids, Anaqueles)
            
            state = env.reset()
            episode_reward = 0
            episode_losses = []
            
            for step in range(MAX_STEPS):
                # Get action using epsilon-greedy policy
                action = get_action(state, epsilon, q_net, env)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                
                # Store experience in replay buffer
                agent.add_experience(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                # Train network
                loss = agent.train_step()
                if loss > 0:
                    episode_losses.append(loss)
                
                if done:
                    break
            
            # Decay exploration rate with minimum bound
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            
            # Log metrics
            rewards_history.append(episode_reward)
            env.render()
            if episode_losses:
                avg_loss = np.mean(episode_losses)
                loss_history.append(avg_loss)
                print(f"Episode {ep+1}/{EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.6f} | Epsilon: {epsilon:.3f}")
            else:
                print(f"Episode {ep+1}/{EPISODES} | Reward: {episode_reward:.2f} | No updates | Epsilon: {epsilon:.3f}")
            
            # Periodically evaluate agent performance
            if (ep + 1) % EVAL_FREQ == 0:
                # Evaluate agent with greedy policy
                eval_reward = evaluate_agent(agent, grupos_df)
                eval_rewards_history.append(eval_reward)
                print(f"Evaluation at episode {ep+1}: Mean reward = {eval_reward:.2f}")
                
                # Save model if performance improved
                if eval_reward > best_eval_reward + MIN_DELTA:
                    best_eval_reward = eval_reward
                    q_net.save_weights(f'results/best_model.weights.h5')
                    print(f"New best model saved with eval reward: {best_eval_reward:.2f}")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                # Early stopping check
                if no_improvement_count >= PATIENCE:
                    print(f"Early stopping triggered after {PATIENCE} evaluations without improvement.")
                    break
                
                # Plot and save progress
                plt.figure(figsize=(15, 5))
                
                # Plot training rewards
                plt.subplot(1, 3, 1)
                plt.plot(rewards_history)
                plt.title('Training Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                
                # Plot evaluation rewards
                plt.subplot(1, 3, 2)
                plt.plot(range(EVAL_FREQ, ep+2, EVAL_FREQ), eval_rewards_history)
                plt.title('Evaluation Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Mean Eval Reward')
                
                # Plot loss
                if loss_history:
                    plt.subplot(1, 3, 3)
                    plt.plot(loss_history)
                    plt.title('Training Loss')
                    plt.xlabel('Update Step')
                    plt.ylabel('Loss')
                
                plt.tight_layout()
                plt.savefig(f'results/training_progress_ep{ep+1}.png')
                plt.close()
    
    # Load best model for final evaluation
    if os.path.exists('results/best_model.weights.h5'):
        q_net.load_weights('results/best_model.weights.h5')
        print("Loaded best model for final evaluation")
    
    # Final comprehensive evaluation
    final_rewards = []
    for grupo_idx in range(len(grupos_df)):
        productos = grupos_df[grupo_idx]
        ids = productos[['PRODUCTO']].to_dict('records')
        Anaqueles = productos[['ANAQUEL']].to_dict('records')
        productos = productos[['ALTO', 'ANCHO', 'LARGO', 'VOLUMEN', 'PESO', "UNDESTIMADAS"]].to_dict('records')
        env = ProductPlacementEnv(productos, ids)
        
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Use greedy policy
            grid = tf.convert_to_tensor(state['grid'][None, ...], dtype=tf.float32)
            product = tf.convert_to_tensor(state['product'][None, ...], dtype=tf.float32)
            q_values = q_net((grid, product), training=False)[0].numpy()
            
            # Only consider valid actions
            mask = (state['grid'].reshape(-1) == 0)
            masked_q = np.copy(q_values)
            masked_q[~mask] = -np.inf
            action = int(np.argmax(masked_q))
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            
            
        final_rewards.append(episode_reward)
        print(f"Final evaluation on group {grupo_idx+1}: Reward = {episode_reward:.2f}")
    
    print(f"Final evaluation average reward: {np.mean(final_rewards):.2f}")
    
    # Save final model
    q_net.save_weights('results/final_model.weights.h5')
    
    # Plot final training curves
    plt.figure(figsize=(15, 5))
    
    # Plot training rewards
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot evaluation rewards
    plt.subplot(1, 3, 2)
    eval_episodes = range(EVAL_FREQ, len(rewards_history) + 1, EVAL_FREQ)
    plt.plot(eval_episodes[:len(eval_rewards_history)], eval_rewards_history)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Mean Eval Reward')
    
    # Plot loss
    if loss_history:
        plt.subplot(1, 3, 3)
        plt.plot(loss_history)
        plt.title('Training Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('results/final_training_summary.png')
    
    print("Training complete!")
    print(env.last_render())