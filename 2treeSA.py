import math
import random
import time
import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Output file name
OUTPUT_FILE = r"C:/Users/user/Downloads/submission (126).csv"

# Optimization Settings
# Lower iterations for N=1-200 to finish in reasonable time. 
# Increase to 50,000+ for competition-grade results.
N_ITERATIONS = 500000   
START_TEMP = 1.5
END_TEMP = 0.001
COOLING_RATE = 0.9995

# Lattice Initial "Safe" State (Wide spacing to ensure no overlaps at start)
INIT_PARAMS = {
    'v1_len': 2.5,       # Column spacing
    'v1_angle': 0.0,     # Row angle
    'v2_len': 2.5,       # Row spacing
    'v2_angle': 60.0,    # Hexagonal-ish start
    'int_x': 1.25,       # Unit cell offset X
    'int_y': 1.25,       # Unit cell offset Y
    'int_rot': 180.0     # Unit cell rotation (flip)
}

# =============================================================================
# 2. GEOMETRY ENGINE
# =============================================================================

def get_base_tree_polygon():
    """Returns the base Santa 2025 tree polygon centered at 0,0."""
    coords = np.array([
        [0.0, 0.8], [0.125, 0.5], [0.0625, 0.5], [0.2, 0.25], [0.1, 0.25], 
        [0.35, 0.0], [0.075, 0.0], [0.075, -0.2], [-0.075, -0.2], [-0.075, 0.0], 
        [-0.35, 0.0], [-0.1, 0.25], [-0.2, 0.25], [-0.0625, 0.5], [-0.125, 0.5]
    ])
    return Polygon(coords)

# Global constant for workers
BASE_POLY = get_base_tree_polygon()

def generate_lattice(params, n_trees):
    """
    Generates N tree positions based on 7 lattice parameters.
    Uses a Metamodel of 2 trees per cell.
    """
    v1_rad = math.radians(params['v1_angle'])
    v2_rad = math.radians(params['v2_angle'])
    
    # Basis Vectors
    v1_x = params['v1_len'] * math.cos(v1_rad)
    v1_y = params['v1_len'] * math.sin(v1_rad)
    v2_x = params['v2_len'] * math.cos(v2_rad)
    v2_y = params['v2_len'] * math.sin(v2_rad)
    
    trees = []
    
    # Grid Dimensions
    # We pack 2 trees per cell, so we need ceil(N/2) cells.
    # We arrange these cells in a roughly square aspect ratio.
    n_cells = math.ceil(n_trees / 2)
    side = math.ceil(math.sqrt(n_cells))
    
    # Generate Grid
    count = 0
    # Create a slightly larger grid and clip to N
    for row in range(side + 1):
        for col in range(side + 1):
            if count >= n_trees: break
            
            # -- Cell Origin --
            cell_x = col * v1_x + row * v2_x
            cell_y = col * v1_y + row * v2_y
            
            # -- Tree 1 (Fixed in cell) --
            # Create dict first, simpler than Polygon object overhead for SA?
            # No, we need Polygon for collision check.
            t1_poly = affinity.translate(BASE_POLY, xoff=cell_x, yoff=cell_y)
            trees.append({'x': cell_x, 'y': cell_y, 'deg': 0.0, 'poly': t1_poly})
            count += 1
            if count >= n_trees: break
            
            # -- Tree 2 (Relative in cell) --
            t2_x = cell_x + params['int_x']
            t2_y = cell_y + params['int_y']
            t2_rot = params['int_rot']
            
            # Rotate then translate
            # Note: rotate around (0,0) of the tree, then move
            t2_poly = affinity.rotate(BASE_POLY, t2_rot, origin=(0,0))
            t2_poly = affinity.translate(t2_poly, xoff=t2_x, yoff=t2_y)
            
            trees.append({'x': t2_x, 'y': t2_y, 'deg': t2_rot, 'poly': t2_poly})
            count += 1
            
    return trees

def check_overlap(trees):
    """Checks for any collision between trees."""
    # Optimization: In a lattice, we mostly care about neighbors.
    # But for robustness, we check all pairs within reasonable distance.
    polys = [t['poly'] for t in trees]
    n = len(polys)
    
    # Simple brute force is fast enough for N=200 with Shapely (~5ms)
    # Could use STRtree for N>500
    for i in range(n):
        for j in range(i + 1, n):
            # Fast dist check (max diam approx 1.6)
            # squared dist > 4.0 means no collision possible
            dx = trees[i]['x'] - trees[j]['x']
            dy = trees[i]['y'] - trees[j]['y']
            if dx*dx + dy*dy > 4.0:
                continue
            
            if polys[i].intersects(polys[j]):
                return True
    return False

def get_score(trees):
    """Minimize Max Dimension (Bounding Box Side)."""
    min_x, min_y = 1e9, 1e9
    max_x, max_y = -1e9, -1e9
    
    # We can iterate over vertices or just polygon bounds
    for t in trees:
        minx, miny, maxx, maxy = t['poly'].bounds
        if minx < min_x: min_x = minx
        if miny < min_y: min_y = miny
        if maxx > max_x: max_x = maxx
        if maxy > max_y: max_y = maxy
        
    w = max_x - min_x
    h = max_y - min_y
    return max(w, h)

# =============================================================================
# 3. SA WORKER FUNCTION
# =============================================================================

def perturb_params(params, temp):
    """Mutate parameters based on temperature."""
    new_p = params.copy()
    
    # Which parameter to change?
    key = random.choice(list(params.keys()))
    
    # Scales
    if 'angle' in key or 'rot' in key:
        # Angles change more aggressively at high temp
        sigma = 5.0 * temp
    else:
        # Lengths change delicately
        sigma = 0.2 * temp
        
    new_p[key] += random.gauss(0, sigma)
    
    # Bounds checks (optional but helpful)
    # Don't let vectors become negative or too small
    if new_p.get('v1_len', 1) < 0.5: new_p['v1_len'] = 0.5
    if new_p.get('v2_len', 1) < 0.5: new_p['v2_len'] = 0.5
    
    return new_p

def solve_for_n(n):
    """Runs SA for a specific group size N."""
    # Special case N=1 (Trivial)
    if n == 1:
        return [{'x': 0.0, 'y': 0.0, 'deg': 0.0}]

    # Initialize
    current_params = INIT_PARAMS.copy()
    
    # Ensure start is valid by expanding if needed
    current_trees = generate_lattice(current_params, n)
    while check_overlap(current_trees):
        current_params['v1_len'] += 0.2
        current_params['v2_len'] += 0.2
        current_trees = generate_lattice(current_params, n)
        
    current_score = get_score(current_trees)
    best_params = current_params.copy()
    best_score = current_score
    best_trees = current_trees
    
    temp = START_TEMP
    
    # Run Loop
    for _ in range(N_ITERATIONS):
        cand_params = perturb_params(current_params, temp)
        cand_trees = generate_lattice(cand_params, n)
        
        # Hard Constraint: Overlap
        if check_overlap(cand_trees):
            # Reject immediately
            # (Or apply massive penalty, but rejection is cleaner for lattice)
            temp *= COOLING_RATE
            continue
            
        cand_score = get_score(cand_trees)
        delta = cand_score - current_score
        
        # Metropolis
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_params = cand_params
            current_score = cand_score
            current_trees = cand_trees
            
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
                best_trees = current_trees
        
        temp *= COOLING_RATE
        if temp < END_TEMP:
            break
            
    # Return formatted list for this group
    return best_trees

# =============================================================================
# 4. MAIN RUNNER
# =============================================================================

def process_wrapper(n):
    """Wrapper to handle seed and return ID-tagged results."""
    # Reseed random for each process to ensure diversity
    random.seed(time.time() + n)
    
    trees = solve_for_n(n)
    
    results = []
    for i, t in enumerate(trees):
        # Format ID: 001_0, 050_1, 200_199
        group_id = f"{n:03d}_{i}"
        
        results.append({
            'id': group_id,
            'x': t['x'],
            'y': t['y'],
            'deg': t['deg']
        })
    return results

if __name__ == "__main__":
    print(f"--- Lattice SA Optimizer (N=1 to 200) ---")
    print(f"Iterations per group: {N_ITERATIONS}")
    print(f"CPU Cores: {cpu_count()}")
    
    all_rows = []
    
    # Create list of tasks (1 to 200)
    tasks = list(range(1, 201))
    
    # Use Multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        # Use tqdm to show progress bar across the 200 tasks
        for result in tqdm(pool.imap(process_wrapper, tasks), total=len(tasks), desc="Optimizing Groups"):
            all_rows.extend(result)
            
    # Save Results
    df = pd.DataFrame(all_rows)
    
    # Optional: Add 's' prefix if strictly required by your specific parser
    # df['x'] = 's' + df['x'].astype(str)
    # df['y'] = 's' + df['y'].astype(str)
    # df['deg'] = 's' + df['deg'].astype(str)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nCompleted! Saved {len(df)} trees to {OUTPUT_FILE}")