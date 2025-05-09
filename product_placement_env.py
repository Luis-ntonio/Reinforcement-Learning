import numpy as np
import gym
from gym import spaces

class ProductPlacementEnv(gym.Env):
    def __init__(self, products, rows=20, cols=20):
        super(ProductPlacementEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.total_cells = rows * cols
        self.products = products
        self.current_index = 0
        self.grid = np.zeros((rows, cols), dtype=np.float32)  # 0: empty, >0: occupied with product quantity
        
        # Center points for different types of distance calculations
        self.center_row = rows // 2
        self.center_col = cols // 2
        
        # Track the cumulative value of products placed
        self.total_value_placed = 0
        self.max_possible_value = sum([p['UNDESTIMADAS'] for p in products])
        
        # === Observation space ===
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(rows, cols, 1), dtype=np.float32),
            'product': spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)  # quantity, volume, height, width, length, weight
        })

        
        # === Action space ===
        self.action_space = spaces.Discrete(self.total_cells)
        
        # Track placed products for better reward calculation
        self.placement_history = []
        
        # For adjacency and product grouping rewards
        self.product_categories = self._categorize_products()
    
    def _categorize_products(self):
        """Categorize products based on their features for adjacency rewards"""
        categories = {}
        
        # Simple categorization based on size
        for i, product in enumerate(self.products):
            # Create a basic category based on volume and weight
            if product['VOLUMEN'] > 0.7:  # High volume
                cat = 'large'
            elif product['VOLUMEN'] > 0.3:  # Medium volume
                cat = 'medium'
            else:  # Small volume
                cat = 'small'
                
            # Add weight modifier
            if product['PESO'] > 0.7:
                cat += '_heavy'
            elif product['PESO'] > 0.3:
                cat += '_medium_weight'
            else:
                cat += '_light'
                
            categories[i] = cat
            
        return categories
    
    def reset(self):
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.current_index = 0
        self.total_value_placed = 0
        self.placement_history = []
        return self._get_obs()
    
    def _get_obs(self):
        grid_obs = self.grid[:, :, None]  # Add channel dimension for CNN
        product = self._get_current_product()
        return {'grid': grid_obs, 'product': product}
    
    def _get_current_product(self):
        if self.current_index >= len(self.products):
            return np.zeros(6, dtype=np.float32)
        
        p = self.products[self.current_index]
        # Combine product attributes into a feature vector
        return np.array([
            p['UNDESTIMADAS'],  # Quantity
            p['VOLUMEN'],       # Volume
            p['ALTO'],          # Height
            p['ANCHO'],         # Width
            p['LARGO'],         # Length
            p['PESO']           # Weight
        ], dtype=np.float32)
    
    def _calculate_only_center_rewards(self, row, col, quantity, volume):
        """
        Recompensa simple basada solo en la distancia:
        borzones (distance_score<0.5) son penalizados,
        el centro (distance_score>0.5) es premiado.
        """
        dist_to_center = np.sqrt((row - self.center_row)**2 + (col - self.center_col)**2)
        max_dist       = np.sqrt(self.rows**2 + self.cols**2) / 2
        distance_score = 1.0 - (dist_to_center / max_dist)        # en [0,1]

        # Nueva función de recompensa lineal centrada en 0.5:
        #   si distance_score=0.5 → reward=0
        #   si distance_score=1.0 → reward=+5
        #   si distance_score=0.0 → reward=−5
        reward = (distance_score - 0.5) * 10.0

        # como debug, devolvemos solo este componente
        return reward, {'center_only': reward}

    def _calculate_improved_rewards(self, row, col, quantity, volume):
        """Calculate more informative and stable rewards"""
        rewards = {}
        
        # === Base placement reward ===
        # Simple reward for successfully placing a product
        rewards['base_placement'] = 0.1
        
        # === Distance-based reward component ===
        # Calculate Euclidean distance to center (optimal distance is near center)
        dist_to_center = np.sqrt((row - self.center_row)**2 + (col - self.center_col)**2)
        max_dist = np.sqrt((self.rows)**2 + (self.cols)**2) / 2
        
        # Normalize and invert (closer = higher reward)
        distance_score = 1.0 - (dist_to_center / max_dist)
        rewards['center_proximity'] = (1 + distance_score * 2)**2
        
        # === Product value reward ===
        # Higher value/quantity products should get better positions
        value_ratio = quantity / max(1.0, self.max_possible_value)
        rewards['value'] = value_ratio * 0.5
        
        # === Adjacency and grouping rewards ===
        adjacency_score = 0
        current_category = self.product_categories.get(self.current_index, '')
        
        # Check all surrounding cells (including diagonals)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        similar_adjacent = 0
        different_adjacent = 0
        empty_adjacent = 0
        
        for dr, dc in directions:
            adj_row, adj_col = row + dr, col + dc
            
            # Skip if outside grid
            if (adj_row < 0 or adj_row >= self.rows or 
                adj_col < 0 or adj_col >= self.cols):
                continue
                
            if self.grid[adj_row, adj_col] == 0:
                empty_adjacent += 1
                continue
            
            # Find which product is in this cell
            product_idx = None
            for placement in self.placement_history:
                if placement['position'] == (adj_row, adj_col):
                    product_idx = placement['product_idx']
                    break
            
            if product_idx is None:
                continue
                
            # Check if adjacent product is in same category
            if self.product_categories.get(product_idx, '') == current_category:
                similar_adjacent += 1
            else:
                different_adjacent += 0.5
        
        # Calculate grouping score - reward similar products being adjacent
        if similar_adjacent > 0:
            adjacency_score = similar_adjacent * 0.3
        
        # Add small reward for any adjacency (better than floating alone)
        if different_adjacent > 0:
            adjacency_score += different_adjacent * 0.1
        
        # Slight preference for having some empty adjacent cells (accessibility)
        if empty_adjacent > 0:
            adjacency_score += min(empty_adjacent, 3) * 0.05
            
        rewards['adjacency'] = 0 #adjacency_score
        
        # === Layout balance reward ===
        # Calculate row and column density
        row_density = np.sum(self.grid > 0, axis=1) / self.cols
        col_density = np.sum(self.grid > 0, axis=0) / self.rows
        
        # Calculate variance (lower variance = more balanced layout)
        row_variance = np.var(row_density)
        col_variance = np.var(col_density)
        
        # Higher reward for more balanced layout
        balance_score = 0.3 * np.exp(-2 * row_variance) + 0.3 * np.exp(-2 * col_variance)
        rewards['balance'] = 0 #balance_score
        
        # === Visual appeal reward - discourage zigzag patterns ===
        # Check if this placement creates a visually pleasing pattern
        visual_score = 0
        
        # Reward straight lines and patterns
        # Check if this forms part of a horizontal line
        if col > 0 and col < self.cols - 1:
            if self.grid[row, col-1] > 0 and self.grid[row, col+1] > 0:
                visual_score += 0.2
        
        # Check if this forms part of a vertical line
        if row > 0 and row < self.rows - 1:
            if self.grid[row-1, col] > 0 and self.grid[row+1, col] > 0:
                visual_score += 0.2
        
        rewards['visual'] = 0#visual_score
        
        # === Progress reward ===
        # Reward progress through product placement
        progress = self.current_index / len(self.products)
        rewards['progress'] = progress * 0.2
        
        # Sum up all reward components
        total_reward = sum(rewards.values())
        return total_reward, rewards

    def _calculate_proximity_reward(self, row, col):
        """Calculate reward based on proximity to center and similar products"""
        # Distance to center reward component
        dist_to_center = np.sqrt((row - self.center_row)**2 + (col - self.center_col)**2)
        max_dist = np.sqrt((self.rows)**2 + (self.cols)**2) / 2  # Maximum possible distance
        center_proximity = 1.0 - (dist_to_center / max_dist)  # Higher when closer to center
        
        # Product adjacency reward component - encourage grouping similar products
        adjacency_reward = 0
        current_category = self.product_categories.get(self.current_index, '')
        
        # Check adjacent cells (up, down, left, right)
        adjacency_directions = [
            (row-1, col), (row+1, col), 
            (row, col-1), (row, col+1)
        ]
        
        similar_adjacent = 0
        different_adjacent = 0
        
        for adj_row, adj_col in adjacency_directions:
            # Skip if outside grid
            if (adj_row < 0 or adj_row >= self.rows or 
                adj_col < 0 or adj_col >= self.cols):
                continue
                
            # Skip empty cells
            if self.grid[adj_row, adj_col] == 0:
                continue
            
            # Find which product is in this cell
            product_idx = None
            for place_idx, placement in enumerate(self.placement_history):
                if placement['position'] == (adj_row, adj_col):
                    product_idx = placement['product_idx']
                    break
            
            # Skip if we can't determine which product
            if product_idx is None:
                continue
                
            # Check if adjacent product is in same category
            if self.product_categories.get(product_idx, '') == current_category:
                similar_adjacent += 1
            else:
                different_adjacent += 0.2  # Small reward for any adjacency
        
        # Calculate adjacency reward - more reward for similar products nearby
        if similar_adjacent > 0 or different_adjacent > 0:
            adjacency_reward = (similar_adjacent * 0.3) + (different_adjacent * 0.1)
        
        # Combine rewards (center proximity has higher weight)
        return (center_proximity * 0.7) + (adjacency_reward * 0.3)
    
    def _calculate_balance_reward(self):
        """Calculate reward for balanced shelf usage"""
        if self.current_index == 0:
            return 0.0
            
        # Calculate how evenly distributed products are across the grid
        occupied_cells = np.sum(self.grid > 0)
        if occupied_cells == 0:
            return 0.0
            
        # Calculate row and column usage
        row_usage = np.sum(self.grid > 0, axis=1) / self.cols  # Usage per row
        col_usage = np.sum(self.grid > 0, axis=0) / self.rows  # Usage per column
        
        # Calculate variance of usage (lower is better - more balanced)
        row_variance = np.var(row_usage)
        col_variance = np.var(col_usage)
        
        # Convert to a reward (higher when variance is lower)
        balance_reward = 0.5 * (1.0 / (1.0 + row_variance)) + 0.5 * (1.0 / (1.0 + col_variance))
        
        return min(0.5, balance_reward)  # Cap at 0.5
    
    def step(self, action):
        row, col = divmod(action, self.cols)
        
        # Default values
        reward = 0.0
        done = False
        
        # Check if action is valid (cell is empty)
        if self.grid[row, col] != 0.0:
            # Invalid placement - cell already occupied
            reward = -5.0  # Stronger penalty for invalid placement
            
            # Do not advance to next product since placement failed
        else:
            # Valid placement
            producto = self.products[self.current_index]
            quantity = producto['UNDESTIMADAS']
            volume = producto['VOLUMEN']
            
            # Mark cell as occupied with product quantity
            self.grid[row, col] = quantity
            
            # Track placement for adjacency calculations
            self.placement_history.append({
                'product_idx': self.current_index,
                'position': (row, col),
                'quantity': quantity,
                'volume': volume
            })
            
            self.total_value_placed += quantity
            
            """# Calculate proximity-based reward
            proximity_reward = self._calculate_proximity_reward(row, col)
            
            # Calculate balance-based reward
            balance_reward = self._calculate_balance_reward()"""

            total_reward, reward_components = self._calculate_only_center_rewards(row, col, quantity, volume)
            reward = total_reward
            
            # Calculate product value reward (normalized by max possible)
            value_reward = quantity / max(1.0, np.max([p['UNDESTIMADAS'] for p in self.products]))
            
            # Calculate progress reward
            progress = self.total_value_placed / self.max_possible_value
            progress_reward = progress * 0.5  # Scale factor
            
            # Add base reward for successful placement
            # reward += 1.0
            
            # Move to next product
            self.current_index += 1
        
        # Check if we've placed all products
        done = self.current_index >= len(self.products)
        
        # Additional completion reward
        """if done:
            # Calculate final balance as part of completion bonus
            final_balance = self._calculate_balance_reward() * 2
            
            # Count empty spaces as measure of efficiency
            occupied = np.sum(self.grid > 0)
            total_cells = self.rows * self.cols
            occupancy_rate = occupied / total_cells
            
            # Prefer higher occupancy but not too crowded (sweet spot around 0.7-0.8)
            if occupancy_rate < 0.5:
                efficiency_bonus = occupancy_rate  # Linear up to 0.5
            elif occupancy_rate <= 0.8:
                efficiency_bonus = 0.5 + (occupancy_rate - 0.5) * 2  # Steeper slope to peak at 0.8
            else:
                efficiency_bonus = 1.0 - (occupancy_rate - 0.8) * 5  # Penalty for overcrowding
            completion_reward = 3.0 + final_balance + efficiency_bonus
            reward += completion_reward"""

        done = (self.current_index >= len(self.products))
        # bonus si terminas de colocar todo
        if done:
            reward += 5.0
        
        return self._get_obs(), reward, done, {
                "reward_components": reward_components,
                "product_placed": self.current_index - 1,
            } 
    
    def render(self, mode='human'):
        """Display the current grid state"""
        np.set_printoptions(precision=2, suppress=True, linewidth=200)
        print("\nCurrent grid state:")
        print(self.grid)
        
        if self.current_index < len(self.products):
            print(f"\nNext product: {self.current_index + 1}/{len(self.products)}")
            p = self.products[self.current_index]
            print(f"Quantity: {p['UNDESTIMADAS']:.2f}, Volume: {p['VOLUMEN']:.2f}")
            print(f"Dimensions: {p['ALTO']:.2f}x{p['ANCHO']:.2f}x{p['LARGO']:.2f}, Weight: {p['PESO']:.2f}")
        else:
            print("\nAll products placed!")
            
        # Display some metrics about the current placement
        occupied = np.sum(self.grid > 0)
        print(f"Occupied cells: {occupied}/{self.rows * self.cols} ({occupied/(self.rows * self.cols):.1%})")
        print(f"Value placed: {self.total_value_placed:.2f}/{self.max_possible_value:.2f} ({self.total_value_placed/self.max_possible_value:.1%})")