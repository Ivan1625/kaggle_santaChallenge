import pandas as pd
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# ==========================================
# 1. GEOMETRY DEFINITIONS (Exact Floats)
# ==========================================
def get_tree_polygon():
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
    return Polygon(coords)

BASE_TREE = get_tree_polygon()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def create_placed_tree(x, y, deg):
    """Creates the polygon at the specific location with high precision."""
    # Rotate around (0,0) then translate
    # Shapely uses standard float64 (double precision)
    rotated = affinity.rotate(BASE_TREE, deg, origin=(0, 0), use_radians=False)
    placed = affinity.translate(rotated, float(x), float(y))
    return placed

def get_puzzle_score_fast(polygons, n):
    """
    Computes the (MaxDim^2)/N score.
    Uses pure float comparisons for speed and precision.
    """
    if not polygons: return 0.0
    
    # Initialize with worst-case floats
    minx, miny = 1e99, 1e99
    maxx, maxy = -1e99, -1e99
    
    for p in polygons:
        # p.bounds returns a tuple of 4 floats
        b = p.bounds
        if b[0] < minx: minx = b[0]
        if b[1] < miny: miny = b[1]
        if b[2] > maxx: maxx = b[2]
        if b[3] > maxy: maxy = b[3]
            
    width = maxx - minx
    height = maxy - miny
    side = max(width, height)
    return (side ** 2) / n

def optimize_puzzle_rotation(df_puzzle, puzzle_id):
    """
    Rotates the entire solution to align it perfectly with the XY axes,
    minimizing the bounding box square.
    """
    n = len(df_puzzle)
    
    # 1. Build initial geometry
    trees = []
    for _, row in df_puzzle.iterrows():
        trees.append(create_placed_tree(row['x'], row['y'], row['deg']))
    
    multi_poly = MultiPolygon(trees)
    # Use centroid as the pivot point for global rotation
    centroid = multi_poly.centroid
    ox, oy = centroid.x, centroid.y

    # 2. Find Rough Angle (Minimum Rotated Rectangle)
    # This gives us a good starting guess
    mrr = multi_poly.minimum_rotated_rectangle
    # Extract coords to find orientation of the rectangle's long axis
    rect_coords = list(mrr.exterior.coords)
    dx = rect_coords[1][0] - rect_coords[0][0]
    dy = rect_coords[1][1] - rect_coords[0][1]
    rough_angle_deg = np.degrees(np.arctan2(dy, dx))
    
    # We check 4 cardinal directions relative to that rectangle
    candidates = [0, -rough_angle_deg, -rough_angle_deg + 90, -rough_angle_deg - 90]
    
    best_rough_angle = 0.0
    best_score = get_puzzle_score_fast(trees, n) # Start with current score

    # Quick check of rough candidates
    for ang in candidates:
        rotated_trees = [affinity.rotate(t, ang, origin=centroid) for t in trees]
        score = get_puzzle_score_fast(rotated_trees, n)
        if score < best_score:
            best_score = score
            best_rough_angle = ang

    # 3. High-Precision Fine Tuning
    # We define the cost function for the optimizer
    def objective(deg_offset):
        # Rotate all trees by deg_offset
        r_trees = [affinity.rotate(t, deg_offset, origin=centroid) for t in trees]
        return get_puzzle_score_fast(r_trees, n)

    # Use 'bounded' optimization to search strictly around the best rough angle.
    # xatol=1e-20 ensures NO ROUNDING (stops only at machine precision limits).
    res = minimize_scalar(
        objective, 
        bounds=(best_rough_angle - 10, best_rough_angle + 10), 
        method='bounded',
        options={'xatol': 1e-20, 'maxiter': 5000} 
    )
    
    final_best_angle = res.x
    final_best_score = res.fun
    
    # 4. Apply the best transformation if it improved
    # We perform the rotation math manually here to ensure we capture the floats directly
    initial_score = get_puzzle_score_fast(trees, n)
    
    if final_best_score < initial_score:
        new_data = []
        rad = np.radians(final_best_angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        
        for i in range(len(trees)):
            old_row = df_puzzle.iloc[i]
            px, py = float(old_row['x']), float(old_row['y'])
            
            # Rotate point (px, py) around (ox, oy)
            nx = ox + cos_a * (px - ox) - sin_a * (py - oy)
            ny = oy + sin_a * (px - ox) + cos_a * (py - oy)
            
            ndeg = float(old_row['deg']) + final_best_angle
            
            new_data.append({
                'id': old_row['id'],
                'x': nx,
                'y': ny,
                'deg': ndeg
            })
        return pd.DataFrame(new_data), initial_score, final_best_score
        
    return df_puzzle, initial_score, initial_score

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    # --- CONFIGURATION ---
    input_file = 'C:/Users/user/Downloads/submission (8).csv'
    output_file = 'C:/Users/user/Downloads/submission_8_rotated.csv'
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # === CLEAN DATA (Remove 's' prefix for math) ===
    # We strip 's' to do math, but we will add it back later
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
    total_init = 0.0
    total_new = 0.0
    results = []
    
    print(f"Optimizing {len(unique_puzzles)} puzzles with machine precision...")
    
    pbar = tqdm(unique_puzzles)
    for n in pbar:
        puzzle_df = df[df['puzzle_n'] == n].copy()
        
        new_df, init_s, new_s = optimize_puzzle_rotation(puzzle_df, n)
        
        total_init += init_s
        total_new += new_s
        results.append(new_df)
        
        if new_s < init_s:
            pbar.set_description(f"Total Improv: {total_init - total_new:.4f}")
    
    # Combine results
    final_df = pd.concat(results).sort_values('id')
    final_df = final_df[['id', 'x', 'y', 'deg']]
    
    # === FORMAT OUTPUT (Add 's' prefix and No Rounding) ===
    print("Formatting output with 's' prefix...")
    
    # We apply the 's' prefix and force high precision string conversion
    for col in ['x', 'y', 'deg']:
        # '%.20f' keeps 20 decimal places, ensuring no precision loss from float64
        final_df[col] = final_df[col].apply(lambda x: f"s{x:.20f}")

    print(f"\n================ RESULTS ================")
    print(f"Original Score: {total_init:.6f}")
    print(f"New Score:      {total_new:.6f}")
    print(f"Improvement:    {total_init - total_new:.6f}")
    
    final_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()