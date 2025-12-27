import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import affinity
import random
from tqdm import tqdm

# =============================================================================
# 1. PARAMETERS
# =============================================================================

# Density Control: 2.0 = Loose (Non-compact, non-dispersed)
SPACING_FACTOR = 2.0  

# Effort Control: Tries per tree before expanding box size
MAX_ATTEMPTS_PER_TREE = 500  

# Fallback Control: Box expansion percentage (0.05 = 5%)
EXPANSION_STEP = 0.05  

# Output File
OUTPUT_FILENAME = "C:/Users/user/Downloads/submission_random.csv"

# =============================================================================
# 2. TREE GEOMETRY
# =============================================================================

def get_tree_polygon(x, y, deg):
    """Creates a Shapely Polygon for a tree at (x,y) with rotation 'deg'."""
    # Standard Tree Vertices (relative to center 0,0)
    base_coords = np.array([
        [0.0, 0.8], [0.125, 0.5], [0.0625, 0.5], [0.2, 0.25], [0.1, 0.25], 
        [0.35, 0.0], [0.075, 0.0], [0.075, -0.2], [-0.075, -0.2], [-0.075, 0.0], 
        [-0.35, 0.0], [-0.1, 0.25], [-0.2, 0.25], [-0.0625, 0.5], [-0.125, 0.5]
    ])
    
    poly = Polygon(base_coords)
    poly = affinity.rotate(poly, deg, origin=(0, 0))
    poly = affinity.translate(poly, xoff=x, yoff=y)
    return poly

def check_collision(new_poly, placed_polys):
    """Checks if new_poly intersects with any polygon in placed_polys."""
    minx, miny, maxx, maxy = new_poly.bounds
    possible_collisions = []
    
    # Broad Phase
    for p in placed_polys:
        p_minx, p_miny, p_maxx, p_maxy = p.bounds
        if (minx > p_maxx or maxx < p_minx or miny > p_maxy or maxy < p_miny):
            continue
        possible_collisions.append(p)
        
    # Narrow Phase
    for p in possible_collisions:
        if new_poly.intersects(p):
            return True
    return False

# =============================================================================
# 3. GENERATION LOGIC
# =============================================================================

def generate_group(n_trees, spacing_factor, max_attempts, expansion_step):
    """Generates a random configuration for a specific N."""
    placed_trees = [] 
    placed_polys = [] 
    
    # Area heuristic
    area_needed = n_trees * 0.7
    current_box_side = np.sqrt(area_needed) * spacing_factor
    half_side = current_box_side / 2
    
    for i in range(n_trees):
        placed = False
        attempts = 0
        
        while not placed:
            attempts += 1
            
            # Generate random candidate
            cx = random.uniform(-half_side, half_side)
            cy = random.uniform(-half_side, half_side)
            deg = random.uniform(0, 360)
            
            new_poly = get_tree_polygon(cx, cy, deg)
            
            if not check_collision(new_poly, placed_polys):
                placed_trees.append({'x': cx, 'y': cy, 'deg': deg})
                placed_polys.append(new_poly)
                placed = True
            
            # If stuck, expand box slightly
            if attempts > max_attempts:
                current_box_side *= (1 + expansion_step)
                half_side = current_box_side / 2
                attempts = 0
                
    return placed_trees

# =============================================================================
# 4. MAIN EXECUTION (1 to 200)
# =============================================================================

if __name__ == "__main__":
    all_results = []
    
    print(f"Starting generation for N=1 to N=200...")
    
    for n in tqdm(range(1, 201), desc="Generating Groups"):
        trees = generate_group(
            n, 
            SPACING_FACTOR, 
            MAX_ATTEMPTS_PER_TREE, 
            EXPANSION_STEP
        )
        
        for i, t in enumerate(trees):
            # Format: '001_0', '020_5', '200_10'
            # :03d ensures 3 digits with leading zeros
            formatted_id = f"{n:03d}_{i}" 
            
            all_results.append({
                'id': formatted_id,
                'x': f"s{t['x']}",    # Added 's' prefix
                'y': f"s{t['y']}",    # Added 's' prefix
                'deg': f"s{t['deg']}" # Added 's' prefix
            })

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\nSuccessfully saved to '{OUTPUT_FILENAME}'")
    
    # --- VISUALIZATION CHECK ---
    try:
        # Check a small number to see the zero padding (e.g., N=5 -> '005')
        sample_n = 5
        print(f"Visualizing sample group N={sample_n} (ID prefix '{sample_n:03d}_')...")
        
        # Filter using the new padded format
        sample_data = df[df['id'].str.startswith(f"{sample_n:03d}_")]
        
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        for _, row in sample_data.iterrows():
            # Strip 's' for plotting
            x = float(row['x'].replace('s', ''))
            y = float(row['y'].replace('s', ''))
            deg = float(row['deg'].replace('s', ''))
            
            poly = get_tree_polygon(x, y, deg)
            px, py = poly.exterior.xy
            ax.fill(px, py, alpha=0.7, fc='forestgreen', ec='black', linewidth=0.5)
            
        plt.axis('equal')
        plt.title(f"Sample N={sample_n}\n(ID format: {sample_n:03d}_...)")
        plt.show()
    except Exception as e:
        print(f"Visualization skipped: {e}")
