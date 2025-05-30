# === Enhanced Hyperparameters ===
EPISODES = 2000  # More episodes for better learning
MAX_STEPS = 200
EPSILON_START = 1.0
EPSILON_END = 0.2  # Lower end value for better exploitation
EPSILON_DECAY = 0.995  # Slower decay for better exploration
GAMMA = 0.99
BATCH_SIZE = 64  # Reduced batch size to stabilize training
REPLAY_CAPACITY = 100000  # Larger buffer for more diverse experiences
LR = 0.00005  # Further reduced learning rate for stability
TAU = 0.005  # Slower target update for stability
REWARD_SCALE = 1  # Lower reward scale to prevent loss explosion
UPDATE_FREQ = 4  # Update network every 4 steps
EVAL_FREQ = 50  # Evaluate and save model more frequently

# Early stopping parameters
PATIENCE = 100  # Number of episodes to wait for improvement
MIN_DELTA = 0.1  # Minimum change to qualify as improvement
