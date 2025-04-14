import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Replace console.log with logging.info
def console_log(*args):
    message = " ".join(map(str, args))
    logging.info(message)

class AssignmentEnv:
    def __init__(self, df:list[pd.DataFrame], alpha=10.0, beta=0.5, gamma=0.5, delta=1.0, theta = 3.0):
        self.df_length = len(df)
        #print(df[0].describe())
        self.df_iter = 0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.theta = theta
        self.dfs = [d.copy().reset_index(drop=True) for d in df]
        for i, d in enumerate(self.dfs):
            self.dfs[i] = d.copy()  # Ensure we are working on a copy
            self.dfs[i].loc[:, 'UNDESTIMADAS_NORM'] = self.dfs[i]['UNDESTIMADAS_NORM'].apply(lambda x: x if x > 0 else 1)
        self.df = self.dfs[self.df_iter]
        self.weight_matrix = np.array([
            [5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5],
            [5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0],
            [4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3],
            [9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0],
            [9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8],
            [10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5],
            [10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5]
        ])
        self.last_placed_items = 0
        self.weight_matrix /= self.weight_matrix.max()
        self.rows = self.weight_matrix.shape[0]
        self.cols = self.weight_matrix.shape[1]
        self.total_cells = self.rows * self.cols
        self.products = np.full((self.rows, self.cols), -1, dtype=int)
        self.num_products = len(self.df)

    def reset(self):
        self.df = self.dfs[self.df_iter]
        self.num_products = len(self.df)
        self.fail_count = 0
        product_ids = self.df['PRODUCTO_NORM'].values.reshape(-1, 1).astype(np.float32)
        quantities = self.df['UNDESTIMADAS_NORM'].values.reshape(-1, 1).astype(np.float32)
        product_features = np.concatenate([product_ids, quantities], axis=1)

        if product_features.shape[0] < self.total_cells:
            pad_size = self.total_cells - product_features.shape[0]
            padding = np.zeros((pad_size, product_features.shape[1]), dtype=product_features.dtype)
            product_features = np.concatenate([product_features, padding], axis=0)

        matrix_input = self.weight_matrix.astype(np.float32)[..., np.newaxis]

        product_mask = np.zeros(self.total_cells, dtype=np.float32)
        product_mask[:self.num_products] = 1.0

        return product_features, matrix_input, product_mask

    def step(self, action):
        grid = np.full((self.rows, self.cols), -1, dtype=int)
        placed_items = 0
        collisions = 0
        weighted_sales = 0.0
        misplaced_penalty = 0.0
        used_cells = set()

        for i, cell in enumerate(action):
            if i >= self.num_products:
                continue
            if cell < 0 or cell >= self.total_cells:
                continue
            row, col = divmod(cell, self.cols)
            if (row, col) in used_cells:
                collisions += 1
                continue
            used_cells.add((row, col))
            grid[row, col] = i
            placed_items += 1

            norm_und = self.df.loc[i, 'UNDESTIMADAS_NORM']
            weight = self.weight_matrix[row, col]
            weighted_sales += norm_und * weight
            if norm_und < 0.1 and weight > 0.8:
                misplaced_penalty += 1.0

        unplaced_items = self.num_products - placed_items
        console_log(self.num_products, weighted_sales, placed_items, collisions, unplaced_items, misplaced_penalty)
        reward = (
            self.alpha * (weighted_sales / (self.num_products + 1e-6)) +
            self.theta * (placed_items / self.total_cells) -
            self.beta * (collisions / self.total_cells) -
            self.gamma * (unplaced_items / self.total_cells) -
            self.delta * (misplaced_penalty / self.num_products)
        )

        #reward = np.clip(reward, -1.0, 1.0)

        self.last_placed_items = placed_items
        done = True
        return grid, reward, done, self.products, weighted_sales
