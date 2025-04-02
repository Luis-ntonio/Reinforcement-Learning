import numpy as np

class AnaquelEnv:
    def __init__(self, df, rows=3, cols=7):
        self.df = df.copy()
        self.weight_matrix = np.array([
            [5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5, 5.5, 4.7, 3.5, 3.0, 2.0, 1.3, 1.0, 1.0, 1.3, 2.0, 3.0, 3.5, 4.7, 5.5],
            [5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0, 5.0, 4.3, 3.0, 2.7, 1.6, 1.0, 0.7, 0.7, 1.0, 1.6, 2.7, 3.0, 4.3, 5.0],
            [4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3, 4.3, 3.6, 2.5, 2.0, 1.3, 0.7, 0.5, 0.5, 0.7, 1.3, 2.0, 2.5, 3.6, 4.3],
            [9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0, 9.0, 7.7, 6.5, 5.5, 5.0, 4.3, 4.0, 4.0, 4.3, 5.0, 5.5, 6.5, 7.7, 9.0],
            [9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8, 9.8, 8.5, 7.0, 6.0, 5.5, 4.7, 4.3, 4.3, 4.7, 5.5, 6.0, 7.0, 8.5, 9.8],
            [10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5],
            [10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5, 10.5, 9.0, 7.7, 6.5, 6.0, 5.0, 4.7, 4.7, 5.0, 6.0, 6.5, 7.7, 9.0, 10.5]
        ])
        # For simplicity, we assume the weight matrix dimensions match our grid dimensions.
        self.rows = self.weight_matrix.shape[0]
        self.cols = self.weight_matrix.shape[1]
        # Matrices to track placements
        self.avail_matrix = np.zeros(self.weight_matrix.shape)  # 0: free, 1: filled
        self.products_id = np.full(self.weight_matrix.shape, -1)  # -1 means no product

        # Mapping product IDs to indexes for one-hot encoding
        unique_products = df['PRODUCTO'].unique()
        self.product_id_to_index = {pid: idx for idx, pid in enumerate(unique_products)}
        self.num_products = len(unique_products)

        # The state will be represented as a flattened vector of shape:
        # (rows * cols) * (1 + num_products)
        self.state_space = self.rows * self.cols * (1 + self.num_products)
        # The action space: choose a product (from the df rows) and choose a cell (from rows*cols)
        self.action_space = (self.rows * self.cols) * self.num_products

        self.failed_attempts = 0
        self.state_quantities = np.zeros(self.weight_matrix.shape)

        # Calculate the maximum possible total cost for normalization
        self.max_possible_total_cost = self.calculate_max_possible_cost()
        self.max_possible_placement_reward = np.log(np.max(self.df['UNDESTIMADAS']) * np.max(self.weight_matrix))

    def reset(self):
        """Reset environment for a new episode."""
        self.state_quantities = np.zeros(self.weight_matrix.shape)
        self.avail_matrix.fill(0)
        self.products_id.fill(-1)
        self.failed_attempts = 0
        return self.get_state()

    def get_state(self):
        """
        Returns a flattened state.
        For each cell, the first element is the quantity (or 0 if empty),
        and the remaining are a one-hot encoding of the product placed (all zeros if none).
        """
        # Create state array with float16 dtype.
        state = np.zeros((self.rows, self.cols, 1 + self.num_products), dtype=np.float16)
        
        # Clip state_quantities to avoid overflow when converting to float16.
        max_val = np.finfo(np.float16).max
        clipped_quantities = np.clip(self.state_quantities, 0, max_val)
        
        state[:, :, 0] = clipped_quantities.astype(np.float16)
        
        # One-hot channels for product placement remain unchanged.
        for i in range(self.rows):
            for j in range(self.cols):
                pid = self.products_id[i, j]
                if pid != -1:
                    idx = self.product_id_to_index.get(pid, None)
                    if idx is not None:
                        state[i, j, idx + 1] = 1.0
        return state.flatten()


    def step(self, action):
        """
        Maps the action integer into a product selection and a fixed cell coordinate.
        If the chosen cell is empty, the product is placed.
        If the cell is already occupied, a penalty is applied.
        Returns next_state, reward, done.
        """
        total_cells = self.rows * self.cols
        item = action // total_cells
        cell = action % total_cells
        row, col = divmod(cell, self.cols)

        # Get product information from dataframe
        product_id = self.df.iloc[item]['PRODUCTO']
        quantity = self.df.iloc[item]['UNDESTIMADAS']

        if self.avail_matrix[row, col] == 0:
            # Check if product_id is already placed
            if product_id in self.products_id:
                self.failed_attempts += 1
                base_penalty = self.max_possible_placement_reward * 10
                reward = -base_penalty * (1 + 0.1 * self.failed_attempts)
                if self.failed_attempts > 10:
                    done = True
                    reward = -base_penalty * 20
                    return self.get_state(), reward, done
            else:
                self.failed_attempts = 0
                self.products_id[row, col] = product_id
                self.state_quantities[row, col] = quantity
                self.avail_matrix[row, col] = 1
                reward = -np.log(quantity * self.weight_matrix[row, col])
        else:
            reward = -self.max_possible_placement_reward * 8  # Penalty for occupied cell

        done = self.is_done()
        if done:
            print("All products placed. Episode complete.")
            reward += 100 - (np.sum(self.state_quantities * self.weight_matrix) / self.max_possible_total_cost)
        next_state = self.get_state()
        return next_state, reward, done

    def calculate_max_possible_cost(self):
        """
        Calculate the maximum possible total cost if all products were placed
        in the worst possible positions.
        """
        sorted_products = self.df.sort_values(by='UNDESTIMADAS', ascending=False)
        flat_weights = self.weight_matrix.flatten()
        sorted_cell_indices = np.argsort(flat_weights)[::-1]
        max_cost = 0
        products_placed = 0
        for i, product in sorted_products.iterrows():
            if products_placed >= len(sorted_cell_indices):
                break
            quantity = product['UNDESTIMADAS']
            cell_index = sorted_cell_indices[products_placed]
            row, col = np.unravel_index(cell_index, self.weight_matrix.shape)
            max_cost += quantity * self.weight_matrix[row, col]
            products_placed += 1
            if products_placed >= min(len(sorted_products), self.rows * self.cols):
                break
        return max_cost

    def is_done(self):
        """Episode is done when all cells are filled."""
        return np.sum(self.avail_matrix) == self.num_products
    