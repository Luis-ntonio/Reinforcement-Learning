import numpy as np
from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class NStepTransitionBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []

    def add(self, transition):
        self.buffer.append(transition)

    def is_ready(self):
        return len(self.buffer) >= self.n_step

    def get_n_step_transition(self):
        R = 0.0
        for i, trans in enumerate(self.buffer):
            R += (self.gamma ** i) * trans.reward
            if trans.done:
                break
        next_state = self.buffer[-1].next_state
        done = self.buffer[-1].done
        state = self.buffer[0].state
        action = self.buffer[0].action
        return Transition(state, action, R, next_state, done)

    def reset(self):
        self.buffer = []