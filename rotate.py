import pandas as pd
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')  # Suppress shapely warnings

# ==========================================
# 1. GEOMETRY DEFINITIONS WITH SCALING
# ==========================================
# Scale factor to avoid floating point errors
SCALE_FACTOR = 1e5  # One million times larger

def get_tree_polygon(scaled=True):
    """Get the base tree polygon, optionally scaled."""
    # Exact coordinates from competition spec
    coords = [
        (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5),
        (0.2, 0.25), (0.1, 0.25),
        (0.35, 0.0), (0.075, 0.0),
        (0.075, -0.2), (-0.075, -0.2),
        (-0.075, 0.0), (-0.35, 0.0),
        (-0.1, 0.25), (-0.2, 0.25),
        (-0.0625, 0.5), (-0.125, 0.5),
        (0.0, 0.8)
    ]
    
    if scaled:
        scaled_coords = [(x * SCALE_FACTOR, y * SCALE_FACTOR) for x, y in coords]
        return Polygon(scaled_coords)
    else:
        return Polygon(coords)

# Pre-compute the base trees
BASE_TREE_SCALED = get_tree_polygon(scaled=True)
BASE_TREE_UNSCALED = get_tree_polygon(scaled=False)

# ==========================================
# 2. HELPER FUNCTIONS WITH SCALING
# ==========================================
def create_placed_tree(x, y, deg, scaled=True):
    """Creates the polygon at the specific location with high precision."""
    if scaled:
        # Scale up coordinates
        x_scaled = float(x) * SCALE_FACTOR
        y_scaled = float(y) * SCALE_FACTOR
        
        # Rotate around (0,0) then translate
        rotated = affinity.rotate(BASE_TREE_SCALED, deg, origin=(0, 0), use_radians=False)
        placed = affinity.translate(rotated, x_scaled, y_scaled)
    else:
        # Use unscaled coordinates directly
        rotated = affinity.rotate(BASE_TREE_UNSCALED, deg, origin=(0, 0), use_radians=False)
        placed = affinity.translate(rotated, float(x), float(y))
        
    return placed

def check_collisions(trees, scaled=True):
    """
    Fast collision checking with early exit.
    
    Args:
        trees: List of tree polygons
        scaled: Whether the trees are already scaled
    """
    # Buffer size depends on whether we're in scaled space
    epsilon = 1e-10 if not scaled else 1e-10 * SCALE_FACTOR
    
    # Check each pair of trees
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if trees[i].intersects(trees[j]):
                # For very close trees, do a more precise check
                if trees[i].buffer(-epsilon).intersects(trees[j].buffer(-epsilon)):
                    return True
    return False

def get_puzzle_score_fast(polygons, n, scaled=True):
    """
    Computes the (MaxDim^2)/N score.
    
    Args:
        polygons: List of tree polygons
        n: Number of trees
        scaled: Whether the polygons are already scaled
    """
    if not polygons: return 0.0
    
    # Initialize with worst-case values
    minx, miny = float('inf'), float('inf')
    maxx, maxy = float('-inf'), float('-inf')
    
    # Fast bounds extraction
    for p in polygons:
        b = p.bounds
        minx = min(minx, b[0])
        miny = min(miny, b[1])
        maxx = max(maxx, b[2])
        maxy = max(maxy, b[3])
            
    side = max(maxx - minx, maxy - miny)
    
    # Scale back down for the score if needed
    if scaled:
        side = side / SCALE_FACTOR
        
    return (side ** 2) / n

def optimize_puzzle_rotation(puzzle_data):
    """
    Optimizes a single puzzle's rotation with scaling.
    
    Args:
        puzzle_data: Tuple of (puzzle_number, dataframe)
    """
    n, df_puzzle = puzzle_data
    
    # 1. Build initial geometry (only once)
    trees = []
    for _, row in df_puzzle.iterrows():
        trees.append(create_placed_tree(row['x'], row['y'], row['deg'], scaled=True))
    
    # Store original for verification
    original_df = df_puzzle.copy()
    initial_score = get_puzzle_score_fast(trees, len(trees), scaled=True)
    
    # Verify no initial collisions
    if check_collisions(trees, scaled=True):
        print(f"Warning: Puzzle {n} has initial collisions!")
        return original_df, initial_score, initial_score
    
    # Create a MultiPolygon for faster operations
    multi_poly = MultiPolygon(trees)
    
    # Calculate centroid once (in scaled coordinates)
    centroid = multi_poly.centroid
    ox, oy = centroid.x, centroid.y
    
    # 2. Find Rough Angle (Minimum Rotated Rectangle)
    mrr = multi_poly.minimum_rotated_rectangle
    rect_coords = list(mrr.exterior.coords)
    dx = rect_coords[1][0] - rect_coords[0][0]
    dy = rect_coords[1][1] - rect_coords[0][1]
    rough_angle_deg = np.degrees(np.arctan2(dy, dx))
    
    # We check 4 cardinal directions relative to that rectangle
    candidates = [0, -rough_angle_deg, -rough_angle_deg + 90, -rough_angle_deg - 90]
    
    best_rough_angle = 0.0
    best_score = initial_score
    
    # Quick check of rough candidates
    for ang in candidates:
        rotated_trees = [affinity.rotate(t, ang, origin=(ox, oy)) for t in trees]
        
        if not check_collisions(rotated_trees, scaled=True):
            score = get_puzzle_score_fast(rotated_trees, len(trees), scaled=True)
            if score < best_score:
                best_score = score
                best_rough_angle = ang
    
    # 3. High-Precision Fine Tuning
    final_best_angle = best_rough_angle
    final_best_score = best_score
    
    if best_rough_angle != 0.0:
        # Define the objective function for the optimizer
        def objective(deg_offset):
            # Rotate all trees by deg_offset
            r_trees = [affinity.rotate(t, deg_offset, origin=(ox, oy)) for t in trees]
            
            if check_collisions(r_trees, scaled=True):
                return float('inf')
                
            return get_puzzle_score_fast(r_trees, len(trees), scaled=True)
    
        # Use 'bounded' optimization to search strictly around the best rough angle.
        res = minimize_scalar(
            objective, 
            bounds=(best_rough_angle - 5, best_rough_angle + 5), 
            method='bounded',
            options={'xatol': 1e-8, 'maxiter': 50} 
        )
        
        final_best_angle = res.x
        final_best_score = res.fun
    
    # 4. Apply the best transformation if it improved
    if final_best_score < initial_score and final_best_score != float('inf'):
        # Get original coordinates
        x_vals = df_puzzle['x'].values
        y_vals = df_puzzle['y'].values
        
        # Convert to numpy for vectorized operations
        rad = np.radians(final_best_angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        # Scale for rotation
        ox_unscaled = ox / SCALE_FACTOR
        oy_unscaled = oy / SCALE_FACTOR
        
        # Vectorized rotation in unscaled space
        new_x = ox_unscaled + cos_a * (x_vals - ox_unscaled) - sin_a * (y_vals - oy_unscaled)
        new_y = oy_unscaled + sin_a * (x_vals - ox_unscaled) + cos_a * (y_vals - oy_unscaled)
        new_deg = df_puzzle['deg'].values + final_best_angle
        
        # Create the new dataframe
        new_df = df_puzzle.copy()
        new_df['x'] = new_x
        new_df['y'] = new_y
        new_df['deg'] = new_deg
        
        # Final collision check in unscaled space for verification
        rotated_trees_unscaled = []
        for i in range(len(new_x)):
            rotated_trees_unscaled.append(create_placed_tree(new_x[i], new_y[i], new_deg[i], scaled=False))
        
        if not check_collisions(rotated_trees_unscaled, scaled=False):
            return new_df, initial_score, final_best_score
        else:
            # If there are collisions in unscaled space, revert to original
            print(f"Warning: Puzzle {n} had collisions after rotation. Using original.")
            return original_df, initial_score, initial_score
    
    # Return original if no improvement or collisions
    return original_df, initial_score, initial_score

# ==========================================
# 3. PARALLEL PROCESSING MAIN
# ==========================================
def main():
    # --- CONFIGURATION ---
    input_file = 'C:/Users/user/Downloads/ensemble.csv'
    output_file = 'C:/Users/user/Downloads/ensemble_rotated.csv'
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # === CLEAN DATA ===
    for col in ['x', 'y', 'deg']:
        if df[col].dtype == 'object':
             df[col] = df[col].astype(str).str.replace('s', '', regex=False)
        df[col] = pd.to_numeric(df[col])
        
    # Extract puzzle number
    try:
        df['puzzle_n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    except:
        print("Error: IDs must be in format 'N_index' (e.g., '16_0')")
        return

    unique_puzzles = df['puzzle_n'].unique()
    
    print(f"Optimizing {len(unique_puzzles)} puzzles with scaled precision...")
    print(f"Using scale factor: {SCALE_FACTOR}")
    
    # Prepare data for parallel processing
    puzzle_data = []
    for n in unique_puzzles:
        puzzle_df = df[df['puzzle_n'] == n].copy()
        puzzle_data.append((n, puzzle_df))
    
    # Use parallel processing
    cpu_count = max(1, mp.cpu_count() - 1)  # Leave one core free
    print(f"Using {cpu_count} CPU cores")
    
    results = []
    total_init = 0.0
    total_new = 0.0
    
    with mp.Pool(processes=cpu_count) as pool:
        # Process puzzles in parallel
        for new_df, init_s, new_s in tqdm(pool.imap(optimize_puzzle_rotation, puzzle_data), 
                                          total=len(puzzle_data)):
            total_init += init_s
            total_new += new_s
            results.append(new_df)
    
    # Combine results
    final_df = pd.concat(results).sort_values('id')
    final_df = final_df[['id', 'x', 'y', 'deg']]
    
    # === FORMAT OUTPUT ===
    print("Formatting output...")
    
    # We apply the 's' prefix and force high precision string conversion
    for col in ['x', 'y', 'deg']:
        # Use 20 decimal places for high precision
        final_df[col] = final_df[col].apply(lambda x: f"s{x:.20f}")

    print(f"\n================ RESULTS ================")
    print(f"Original Score: {total_init:.15f}")
    print(f"New Score:      {total_new:.15f}")
    print(f"Improvement:    {total_init - total_new:.15f}")

    # Final verification
    print("Performing final verification...")
    
    # Create unscaled trees for final verification
    all_trees = []
    for _, row in df.iterrows():
        all_trees.append(create_placed_tree(row['x'], row['y'], row['deg'], scaled=False))
    
    if check_collisions(all_trees, scaled=False):
        print("Warning: Original solution has collisions!")
    
    all_trees_final = []
    for _, row in final_df.iterrows():
        x = float(row['x'].replace('s', ''))
        y = float(row['y'].replace('s', ''))
        deg = float(row['deg'].replace('s', ''))
        all_trees_final.append(create_placed_tree(x, y, deg, scaled=False))
    
    final_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
