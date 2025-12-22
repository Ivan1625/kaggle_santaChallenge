import pandas as pd
import numpy as np
import math
import copy
import random
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

# ==========================================
# 1. EXACT KAGGLE GEOMETRY
# ==========================================
def get_tree_polygon():
    # Exact coordinates (Trunk Top at 0,0)
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
# 2. STATE & PHYSICS ENGINE
# ==========================================

class PuzzleState:
    def __init__(self, ids, xs, ys, rots):
        self.ids = ids
        # Force float64 for max precision
        self.xs = xs.astype(np.float64)
        self.ys = ys.astype(np.float64)
        self.rots = rots.astype(np.float64)
        self.polys = self._build_polys()
        self.bounds = self._calc_bounds()
        self.square_side = self._calc_square_side()
        self.valid = self._check_validity()
        
    def _build_polys(self):
        polys = []
        for x, y, rot in zip(self.xs, self.ys, self.rots):
            p = affinity.rotate(BASE_TREE, rot, origin=(0,0), use_radians=False)
            p = affinity.translate(p, x, y)
            polys.append(p)
        return polys
    
    def _calc_bounds(self):
        if not self.polys: return 0,0,0,0
        # Initialize with first poly
        minx, miny, maxx, maxy = self.polys[0].bounds
        for p in self.polys[1:]:
            b = p.bounds
            if b[0] < minx: minx = b[0]
            if b[1] < miny: miny = b[1]
            if b[2] > maxx: maxx = b[2]
            if b[3] > maxy: maxy = b[3]
        return minx, miny, maxx, maxy

    def _calc_square_side(self):
        minx, miny, maxx, maxy = self.bounds
        w = maxx - minx
        h = maxy - miny
        return max(w, h)

    def _check_validity(self):
        """Returns True if NO overlaps exist."""
        n = len(self.polys)
        for i in range(n):
            p1 = self.polys[i]
            # Use a tiny negative buffer (-1e-9) to allow 'kissing' contact
            p1_s = p1.buffer(-1e-9)
            for j in range(i + 1, n):
                p2 = self.polys[j]
                
                # Fast Bound Check
                if (p1.bounds[2] < p2.bounds[0] or p1.bounds[0] > p2.bounds[2] or 
                    p1.bounds[3] < p2.bounds[1] or p1.bounds[1] > p2.bounds[3]):
                    continue
                
                # Intersection check
                if p1_s.intersects(p2):
                    return False
        return True

def resolve_overlaps_force_directed(state, max_steps=200):
    """
    Force-Directed Separation.
    """
    current_xs = state.xs.copy()
    current_ys = state.ys.copy()
    current_polys = state.polys[:] # Copy list of references
    
    resolved = False
    
    # Damping factor
    alpha = 0.5
    
    for _ in range(max_steps):
        has_overlap = False
        moves_x = np.zeros_like(current_xs)
        moves_y = np.zeros_like(current_ys)
        
        n = len(current_polys)
        for i in range(n):
            p1 = current_polys[i]
            p1_s = p1.buffer(-1e-9)
            
            for j in range(i+1, n):
                p2 = current_polys[j]
                
                if (p1.bounds[2] < p2.bounds[0] or p1.bounds[0] > p2.bounds[2] or 
                    p1.bounds[3] < p2.bounds[1] or p1.bounds[1] > p2.bounds[3]):
                    continue
                
                if p1_s.intersects(p2):
                    has_overlap = True
                    # Repulsion
                    c1 = p1.centroid
                    c2 = p2.centroid
                    dx = c1.x - c2.x
                    dy = c1.y - c2.y
                    dist = math.hypot(dx, dy)
                    
                    if dist < 1e-9:
                        # Random kick
                        angle = random.uniform(0, 2*math.pi)
                        dx, dy = math.cos(angle), math.sin(angle)
                        dist = 1.0
                    
                    # Force Magnitude
                    force = 0.1 * alpha
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    
                    moves_x[i] += fx; moves_y[i] += fy
                    moves_x[j] -= fx; moves_y[j] -= fy

        if not has_overlap:
            resolved = True
            break
            
        # Apply moves
        current_xs += moves_x
        current_ys += moves_y
        
        # Rebuild polys (expensive but necessary)
        new_polys = []
        for k in range(len(current_xs)):
            p = affinity.rotate(BASE_TREE, state.rots[k], origin=(0,0), use_radians=False)
            p = affinity.translate(p, current_xs[k], current_ys[k])
            new_polys.append(p)
        current_polys = new_polys
        
    if resolved:
        return PuzzleState(state.ids, current_xs, current_ys, state.rots)
    return None

def compress_packing(state, strength=0.01):
    """Gravity: Pull towards center."""
    minx, miny, maxx, maxy = state.bounds
    cx, cy = (minx+maxx)/2, (miny+maxy)/2
    
    new_xs = state.xs.copy()
    new_ys = state.ys.copy()
    
    for k in range(len(new_xs)):
        dx = cx - new_xs[k]
        dy = cy - new_ys[k]
        new_xs[k] += dx * strength
        new_ys[k] += dy * strength
        
    return PuzzleState(state.ids, new_xs, new_ys, state.rots)

# ==========================================
# 3. OPTIMIZER LOOP
# ==========================================

def optimize_puzzle(df_puzzle, iterations=100):
    """
    Attempts to squash a single puzzle.
    """
    # Initialize Best State from Input
    best_state = PuzzleState(
        df_puzzle['id'].values,
        df_puzzle['x'].values,
        df_puzzle['y'].values,
        df_puzzle['deg'].values
    )
    
    # If input is invalid, we can't squash it safely.
    if not best_state.valid:
        # print(f"Warning: Puzzle {df_puzzle.iloc[0]['id']} input invalid.")
        return df_puzzle, 0.0
    
    current_state = copy.deepcopy(best_state)
    initial_score = best_state.square_side ** 2
    
    # Iteration loop
    for i in range(iterations):
        # 1. Compress (Gravity)
        # Pressure decreases over time
        pressure = 0.05 * (1 - i/iterations) + 0.005
        compressed = compress_packing(current_state, strength=pressure)
        
        # 2. Mutate (Rotation)
        idx = random.randint(0, len(compressed.ids)-1)
        rot_mag = max(0.1, 10.0 * (1 - i/iterations)) 
        compressed.rots[idx] = (compressed.rots[idx] + random.gauss(0, rot_mag)) % 360.0
        
        # Rebuild poly for the mutated item
        # (Lazy rebuild done inside PuzzleState constructor above, so it's fine)
        # Actually we need to force rebuild if we modify inplace, but here we created new object 'compressed'
        
        # 3. Separate (Resolve)
        resolved_state = resolve_overlaps_force_directed(compressed, max_steps=100)
        
        if resolved_state is not None:
            # Valid state found
            current_state = resolved_state
            
            # Is it better?
            if current_state.square_side < best_state.square_side:
                best_state = copy.deepcopy(current_state)
        else:
            # Failed to resolve, discard compression
            pass
            
    # Return result as DF
    res_data = []
    for k in range(len(best_state.ids)):
        res_data.append({
            'id': best_state.ids[k],
            'x': best_state.xs[k],
            'y': best_state.ys[k],
            'deg': best_state.rots[k]
        })
    
    final_score = best_state.square_side ** 2
    improvement = initial_score - final_score
    
    return pd.DataFrame(res_data), improvement

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    # --- CONFIGURATION ---
    input_file = 'C:/Users/user/Downloads/submission (fk rounding).csv'  # <--- INPUT
    output_file = 'C:/Users/user/Downloads/submission_spyrrow.csv' # <--- OUTPUT
    
    # Optimizer settings
    ITERATIONS_PER_PUZZLE = 150
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Clean Data ('s' prefix)
    for col in ['x', 'y', 'deg']:
        if df[col].dtype == 'object':
             df[col] = df[col].astype(str).str.replace('s', '', regex=False)
        df[col] = pd.to_numeric(df[col])
        
    try:
        df['puzzle_n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    except:
        print("Error: IDs must be in 'N_index' format")
        return

    unique_puzzles = df['puzzle_n'].unique()
    
    total_improvement = 0.0
    final_dfs = []
    
    print(f"Optimizing {len(unique_puzzles)} puzzles...")
    print(f"{'Puzzle':<10} | {'Improvement (10 s.f.)':<20} | {'Current Total':<20}")
    print("-" * 60)
    
    pbar = tqdm(unique_puzzles)
    
    for n in pbar:
        puzzle_df = df[df['puzzle_n'] == n].copy()
        
        # Run Optimizer
        optimized_df, improvement = optimize_puzzle(puzzle_df, iterations=ITERATIONS_PER_PUZZLE)
        
        # Accumulate
        total_improvement += improvement
        final_dfs.append(optimized_df)
        
        # Display logic
        if improvement > 1e-12:
            desc = f"Imp: {improvement:.10f}"
        else:
            desc = "No Change"
            
        pbar.set_description(f"Total Imp: {total_improvement:.8f}")
        
    # Combine results
    final_df = pd.concat(final_dfs).sort_values('id')
    final_df = final_df[['id', 'x', 'y', 'deg']]
    
    # Format Output (Add 's', 20 decimal places)
    print("\nFormatting output...")
    for col in ['x', 'y', 'deg']:
        final_df[col] = final_df[col].apply(lambda x: f"s{x:.20f}")
        
    print(f"\n================ FINAL SUMMARY ================")
    print(f"Total Improvement: {total_improvement:.10f}")
    
    final_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()