import numpy as np
import gym
from gym import spaces

class ProductPlacementEnv(gym.Env):
    def __init__(self, products, rows=14, cols=7):
        super(ProductPlacementEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.total_cells = rows * cols
        self.products = products
        self.current_index = 0
        self.grid = np.zeros((rows, cols), dtype=np.float32)  # 0: vacío, 1: ocupado

        # === Observation ===
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(rows, cols, 1), dtype=np.float32),
            'product': spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)  # cantidad, volumen, alto, ancho, largo
        })

        # === Action ===
        self.action_space = spaces.Discrete(self.total_cells)

    def reset(self):
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.current_index = 0
        return self._get_obs()

    def _get_obs(self):
        grid_obs = self.grid[:, :, None]  # Agregamos canal para CNN
        product = self._get_current_product()
        return {'grid': grid_obs, 'product': product}

    def _get_current_product(self):
        if self.current_index >= len(self.products):
            return np.zeros(5, dtype=np.float32)
        p = self.products[self.current_index]
        volume = p['alto'] * p['ancho'] * p['largo']
        return np.array([p['cantidad'], volume, p['alto'], p['ancho'], p['largo']], dtype=np.float32)

    def step(self, action):
        row, col = divmod(action, self.cols)
        reward = 0
        done = False

        if self.grid[row, col] == 1.0:
            reward = -1.0  # penalización por intentar colocar en celda ocupada
        else:
            producto = self.products[self.current_index]
            volume = producto['alto'] * producto['ancho'] * producto['largo']
            self.grid[row, col] = 1.0

            # Centro de la matriz
            center_row, center_col = self.rows // 2, self.cols // 2
            dist = np.linalg.norm([row - center_row, col - center_col])
            reward = 5.0 / (1 + dist)  # más recompensa cerca del centro

            # Bonus si es grande y se coloca arriba
            if row < self.rows // 3 and volume > 1000:
                reward += 2.0

            self.current_index += 1

        done = self.current_index >= len(self.products)
        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(self.grid)
