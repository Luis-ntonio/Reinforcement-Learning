import random
import numpy as np
from collections import deque, namedtuple

Transition = namedtuple('Transition', (
    'state',            # Tuple of (product_features, matrix_input, product_mask)
    'action',           # List of actions (cell assignments)
    'reward',
    'next_state',       # Tuple of (next_product_features, next_matrix_input, next_product_mask)
    'done'
))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        #print(len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

