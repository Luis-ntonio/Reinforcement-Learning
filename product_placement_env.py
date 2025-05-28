import numpy as np
import gym
from gym import spaces

class ProductPlacementEnv(gym.Env):
    def __init__(self, products, ids, rows=7, cols=14):
        super(ProductPlacementEnv, self).__init__()
        self.ids = ids
        self.rows = rows
        self.cols = cols
        self.total_cells = rows * cols
        self.products = products
        self.current_index = 0
        self.grid = np.zeros((rows, cols), dtype=np.float32)  # 0: empty, >0: occupied with product quantity
        self.output_grid = np.zeros((rows, cols), dtype=np.float32)  # 0: empty, >0: occupied with product quantity
        
        # print(f"Product IDs: {self.ids}")
        # Center points for different types of distance calculations
        self.center_row = 1
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
        self.output_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
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
            ids = self.ids[self.current_index]['PRODUCTO']
            
            # Mark cell as occupied with product quantity
            self.grid[row, col] = quantity
            self.output_grid[row, col] = ids
            
            # Track placement for adjacency calculations
            self.placement_history.append({
                'product_idx': self.current_index,
                'position': (row, col),
                'quantity': quantity,
                'volume': volume
            })
            
            self.total_value_placed += quantity * 2
            
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

    def last_render(self, mode='human'):
        """Display the current grid state"""
        np.set_printoptions(precision=2, suppress=True, linewidth=200, formatter={'float': lambda x: f"{x:.2f}"})
        print("\nCurrent grid state:")
        print(self.output_grid)
        
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