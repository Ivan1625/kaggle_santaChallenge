import numpy as np
import pandas as pd
import math
import random
import time
from numba import njit, prange, float64, int64, boolean
from tqdm import tqdm
from shapely import affinity
from shapely.geometry import Polygon

# ==========================================
# 1. GEOMETRY CONSTANTS
# ==========================================
# Exact Santa Tree Vertices
TREE_X = np.array([0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, 
                   -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125, 0.0], dtype=np.float64)
TREE_Y = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, 
                   -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.8], dtype=np.float64)
NUM_VERTS = len(TREE_X)

# ==========================================
# 2. NUMBA GEOMETRY KERNEL (STRICT)
# ==========================================

@njit(cache=True)
def transform_poly(x, y, rot_deg, out_x, out_y):
    """Transforms polygon vertices based on x, y, rotation."""
    rad = np.radians(rot_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    for i in range(NUM_VERTS):
        tx = TREE_X[i] * cos_a - TREE_Y[i] * sin_a
        ty = TREE_X[i] * sin_a + TREE_Y[i] * cos_a
        out_x[i] = tx + x
        out_y[i] = ty + y

@njit(cache=True)
def get_bounds(px, py):
    minx, miny = 1e9, 1e9
    maxx, maxy = -1e9, -1e9
    for i in range(NUM_VERTS):
        if px[i] < minx: minx = px[i]
        if px[i] > maxx: maxx = px[i]
        if py[i] < miny: miny = py[i]
        if py[i] > maxy: maxy = py[i]
    return minx, miny, maxx, maxy

@njit(cache=True)
def is_point_in_poly(px, py, poly_x, poly_y):
    """Ray casting point-in-poly check."""
    inside = False
    j = NUM_VERTS - 2 
    for i in range(NUM_VERTS - 1):
        if ((poly_y[i] > py) != (poly_y[j] > py)):
            if (px < (poly_x[j] - poly_x[i]) * (py - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
                inside = not inside
        j = i
    return inside

@njit(cache=True)
def segments_intersect(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
    """
    Strict segment intersection. 
    Returns True if segments cross or touch within epsilon.
    """
    def cross(ax, ay, bx, by): return ax*by - ay*bx
    r_x = a2x - a1x; r_y = a2y - a1y
    s_x = b2x - b1x; s_y = b2y - b1y
    denom = cross(r_x, r_y, s_x, s_y)
    
    if np.abs(denom) < 1e-12: return False # Parallel
    
    u_numer = cross(b1x - a1x, b1y - a1y, r_x, r_y)
    t_numer = cross(b1x - a1x, b1y - a1y, s_x, s_y)
    
    u = u_numer / denom
    t = t_numer / denom
    
    # Epsilon for strict collision
    # We treat touching (t approx 0 or 1) as collision to be safe
    eps = 1e-9
    if (t > -eps) and (t < 1.0 + eps) and (u > -eps) and (u < 1.0 + eps):
        return True
    return False

@njit(cache=True)
def check_poly_overlap(p1x, p1y, p2x, p2y, safety_margin):
    """
    Checks if two polygons overlap.
    Includes a 'safety_margin' to ensure they stay slightly apart.
    """
    # 1. Bounding Box (with margin)
    minx1, miny1, maxx1, maxy1 = get_bounds(p1x, p1y)
    minx2, miny2, maxx2, maxy2 = get_bounds(p2x, p2y)
    
    # If boxes don't overlap (plus margin), polygons definitely don't
    if (minx1 > maxx2 + safety_margin or maxx1 < minx2 - safety_margin or 
        miny1 > maxy2 + safety_margin or maxy1 < miny2 - safety_margin):
        return False
        
    # 2. Vertices in Polygon
    # Check p1 verts in p2
    for i in range(NUM_VERTS - 1):
        if is_point_in_poly(p1x[i], p1y[i], p2x, p2y): return True
    # Check p2 verts in p1
    for i in range(NUM_VERTS - 1):
        if is_point_in_poly(p2x[i], p2y[i], p1x, p1y): return True
        
    # 3. Edge Intersections
    # This catches "crossing" without vertex containment
    for i in range(NUM_VERTS - 1):
        for j in range(NUM_VERTS - 1):
            if segments_intersect(p1x[i], p1y[i], p1x[i+1], p1y[i+1],
                                  p2x[j], p2y[j], p2x[j+1], p2y[j+1]):
                return True
    return False

@njit(cache=True)
def get_system_score_raw(xs, ys, rots, n, temp_cache_x, temp_cache_y):
    """Returns Side^2 (Area) of the bounding square."""
    g_minx, g_miny = 1e9, 1e9
    g_maxx, g_maxy = -1e9, -1e9
    for i in range(n):
        transform_poly(xs[i], ys[i], rots[i], temp_cache_x[i], temp_cache_y[i])
        minx, miny, maxx, maxy = get_bounds(temp_cache_x[i], temp_cache_y[i])
        if minx < g_minx: g_minx = minx
        if miny < g_miny: g_miny = miny
        if maxx > g_maxx: g_maxx = maxx
        if maxy > g_maxy: g_maxy = maxy
    w = g_maxx - g_minx
    h = g_maxy - g_miny
    side = w if w > h else h
    return side * side

# ==========================================
# 3. STRICT SIMULATED ANNEALING
# ==========================================

@njit(cache=True)
def run_strict_sa(xs, ys, rots, n, iterations, start_temp, cooling_rate, squeeze_factor, squeeze_freq, safety_margin):
    # Pre-allocate polygon caches
    cache_x = np.zeros((n, 16), dtype=np.float64)
    cache_y = np.zeros((n, 16), dtype=np.float64)
    
    # Initialize Geometry
    for i in range(n):
        transform_poly(xs[i], ys[i], rots[i], cache_x[i], cache_y[i])
        
    # Initial Score
    current_area = get_system_score_raw(xs, ys, rots, n, cache_x, cache_y)
    best_area = current_area
    
    # Backups
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_rots = rots.copy()
    
    temp = start_temp
    
    # Calculate Center
    g_minx, g_miny = 1e9, 1e9
    g_maxx, g_maxy = -1e9, -1e9
    for i in range(n):
        minx, miny, maxx, maxy = get_bounds(cache_x[i], cache_y[i])
        if minx < g_minx: g_minx = minx
        if maxx > g_maxx: g_maxx = maxx
        if miny < g_miny: g_miny = miny
        if maxy > g_maxy: g_maxy = maxy
    cx = (g_minx + g_maxx) / 2
    cy = (g_miny + g_maxy) / 2

    # Loop
    for k in range(iterations):
        
        # --- TYPE 1: TRY SQUEEZE (Hydraulic Press) ---
        # Only applied if VALID. Does not force overlaps.
        did_squeeze = False
        if k % squeeze_freq == 0:
            # Try to squeeze ALL items
            saved_xs = xs.copy()
            saved_ys = ys.copy()
            
            squeeze_valid = True
            
            # Update positions
            for i in range(n):
                xs[i] = cx + (xs[i] - cx) * squeeze_factor
                ys[i] = cy + (ys[i] - cy) * squeeze_factor
            
            # Check validity of squeezed state
            # We must update temp geom to check
            for i in range(n):
                transform_poly(xs[i], ys[i], rots[i], cache_x[i], cache_y[i])
            
            # All-vs-All Check
            for i in range(n):
                for j in range(i+1, n):
                    if check_poly_overlap(cache_x[i], cache_y[i], cache_x[j], cache_y[j], safety_margin):
                        squeeze_valid = False
                        break
                if not squeeze_valid: break
            
            if squeeze_valid:
                # Keep squeeze
                did_squeeze = True
                current_area = get_system_score_raw(xs, ys, rots, n, cache_x, cache_y)
                if current_area < best_area:
                    best_area = current_area
                    best_xs = xs.copy()
                    best_ys = ys.copy()
                    best_rots = rots.copy()
            else:
                # Revert squeeze
                xs[:] = saved_xs[:]
                ys[:] = saved_ys[:]
                # Restore cache
                for i in range(n):
                    transform_poly(xs[i], ys[i], rots[i], cache_x[i], cache_y[i])

        # --- TYPE 2: PERTURBATION (Random Move) ---
        # If we didn't squeeze (or squeeze failed), try a standard move
        if not did_squeeze:
            idx = np.random.randint(0, n)
            old_x, old_y, old_rot = xs[idx], ys[idx], rots[idx]
            
            rnd = np.random.random()
            # Dynamic magnitude
            mag = max(0.00000001, temp * 0.05) 
            
            if rnd < 0.5: # Translate
                xs[idx] += np.random.normal(0, mag)
                ys[idx] += np.random.normal(0, mag)
            elif rnd < 0.9: # Rotate
                rots[idx] += np.random.normal(0, mag * 50.0) 
            else: # Small Jump
                if temp > 1.0:
                    xs[idx] += np.random.normal(0, 0.5)
                    ys[idx] += np.random.normal(0, 0.5)
                    rots[idx] = np.random.uniform(0, 360)

            # Temp geometry
            tx = np.zeros(16, dtype=np.float64)
            ty = np.zeros(16, dtype=np.float64)
            transform_poly(xs[idx], ys[idx], rots[idx], tx, ty)
            
            # CHECK VALIDITY (Hard Constraint)
            valid = True
            for j in range(n):
                if idx == j: continue
                # Check against cached others
                if check_poly_overlap(tx, ty, cache_x[j], cache_y[j], safety_margin):
                    valid = False
                    break
            
            if valid:
                # Commit to cache
                cache_x[idx] = tx
                cache_y[idx] = ty
                
                new_area = get_system_score_raw(xs, ys, rots, n, cache_x, cache_y)
                delta = new_area - current_area
                
                # Metropolis
                if delta < 0 or np.random.random() < np.exp(-delta / temp):
                    current_area = new_area
                    if current_area < best_area:
                        best_area = current_area
                        best_xs = xs.copy()
                        best_ys = ys.copy()
                        best_rots = rots.copy()
                else:
                    # Revert choice
                    xs[idx] = old_x
                    ys[idx] = old_y
                    rots[idx] = old_rot
                    transform_poly(old_x, old_y, old_rot, cache_x[idx], cache_y[idx])
            else:
                # Revert invalid
                xs[idx] = old_x
                ys[idx] = old_y
                rots[idx] = old_rot

        # Update Cooling
        temp *= cooling_rate
        if temp < 1e-12: temp = 1e-12

    return best_xs, best_ys, best_rots, best_area

# ==========================================
# 4. WRAPPER & SHAPELY VERIFICATION
# ==========================================

def get_base_shapely_poly():
    X = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, 
         -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125, 0.0]
    Y = [0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, 
         -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.8]
    return Polygon(list(zip(X, Y)))

BASE_SHAPELY = get_base_shapely_poly()

def verify_validity(xs, ys, rots):
    """
    Final Truth Check using Shapely.
    Returns True if valid.
    """
    polys = []
    n = len(xs)
    for i in range(n):
        p = affinity.rotate(BASE_SHAPELY, rots[i], origin=(0,0))
        p = affinity.translate(p, xs[i], ys[i])
        polys.append(p)
    
    for i in range(n):
        # Buffer -1e-9 allows 'kissing' but not overlap
        p1 = polys[i].buffer(-1e-9) 
        for j in range(i+1, n):
            if p1.intersects(polys[j]):
                return False
    return True

def optimize_puzzle_strict(df_puzzle, params):
    n = len(df_puzzle)
    xs = np.ascontiguousarray(df_puzzle['x'].values, dtype=np.float64)
    ys = np.ascontiguousarray(df_puzzle['y'].values, dtype=np.float64)
    rots = np.ascontiguousarray(df_puzzle['deg'].values, dtype=np.float64)
    
    # 1. Check Initial Validity
    if not verify_validity(xs, ys, rots):
        # If input has collisions, we cannot run Strict SA safely.
        # Returning infinity score ensures we don't save this as a "Best"
        return df_puzzle, float('inf')

    # Warmup Numba
    if not hasattr(optimize_puzzle_strict, "compiled"):
        run_strict_sa(xs.copy(), ys.copy(), rots.copy(), n, 10, 0.1, 0.9, 1.0, 100, 1e-7)
        optimize_puzzle_strict.compiled = True
        
    best_xs, best_ys, best_rots, best_raw_area = run_strict_sa(
        xs, ys, rots, n, 
        iterations=params['ITERATIONS'], 
        start_temp=params['START_TEMP'],
        cooling_rate=params['COOLING_RATE'], 
        squeeze_factor=params['SQUEEZE_FACTOR'],
        squeeze_freq=params['SQUEEZE_FREQ'],
        safety_margin=params['SAFETY_MARGIN']
    )
    
    # Final Double Check
    if not verify_validity(best_xs, best_ys, best_rots):
        return df_puzzle, float('inf')
    
    res_data = []
    for i in range(n):
        res_data.append({
            'id': df_puzzle.iloc[i]['id'],
            'x': best_xs[i],
            'y': best_ys[i],
            'deg': best_rots[i]
        })
        
    final_score = best_raw_area / n
    return pd.DataFrame(res_data), final_score

def main():
    # =====================
    # PARAMS
    # =====================
    input_file = 'C:/Users/user/Downloads/sample_submission.csv'
    output_file = 'C:/Users/user/Downloads/submission_no_collision.csv'
    
    TARGET_PUZZLES = [16] # e.g. [16, 17]
    LOOPS_PER_PUZZLE = 10
    
    PARAMS = {
        'ITERATIONS': 200_000,    
        'START_TEMP': 10.0,       
        'COOLING_RATE': 0.99995,  
        'SQUEEZE_FACTOR': 0.9995, 
        'SQUEEZE_FREQ': 100,
        'SAFETY_MARGIN': 1e-20  # Buffer to guarantee no collision
    }
    
    # =====================
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    for col in ['x', 'y', 'deg']:
        if df[col].dtype == 'object':
             df[col] = df[col].astype(str).str.replace('s', '', regex=False)
        df[col] = pd.to_numeric(df[col])
    
    df['puzzle_n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    
    all_puzzles = df['puzzle_n'].unique()
    if TARGET_PUZZLES:
        puzzles_to_run = [p for p in all_puzzles if p in TARGET_PUZZLES]
    else:
        puzzles_to_run = all_puzzles
        
    final_dfs = []
    total_imp_global = 0.0
    
    print(f"Optimizing {len(puzzles_to_run)} puzzles (Strict Mode)...")
    
    for n in tqdm(puzzles_to_run):
        puzzle_df = df[df['puzzle_n'] == n].copy()
        
        # Initial Score
        # We need to calc initial area first
        # (Assuming input is valid, if not, optimize_puzzle_strict returns inf anyway)
        # Just use a dummy call to get score if valid
        try:
            # Quick calc
            n_items = len(puzzle_df)
            xs = puzzle_df['x'].values
            ys = puzzle_df['y'].values
            rots = puzzle_df['deg'].values
            if verify_validity(xs, ys, rots):
                 # Calc score
                 # Reusing the Numba kernel for score calc needs arrays
                 tx = np.zeros((n_items, 16)); ty = np.zeros((n_items, 16))
                 raw_a = get_system_score_raw(xs.astype(np.float64), ys.astype(np.float64), rots.astype(np.float64), n_items, tx, ty)
                 original_score = raw_a / n_items
            else:
                 original_score = float('inf')
                 tqdm.write(f"Puzzle {n} starting invalid (Collisions). Skipping...")
        except:
             original_score = float('inf')

        if original_score == float('inf'):
            final_dfs.append(puzzle_df)
            continue

        best_df_for_n = puzzle_df
        best_score_for_n = original_score
        
        for i in range(LOOPS_PER_PUZZLE):
            opt_df, new_score = optimize_puzzle_strict(puzzle_df.copy(), PARAMS)
            
            if new_score < best_score_for_n:
                best_score_for_n = new_score
                best_df_for_n = opt_df
                # tqdm.write(f"Puzzle {n} Loop {i}: Improved to {new_score:.10f}")
                
        final_dfs.append(best_df_for_n)
        if best_score_for_n < original_score:
            total_imp_global += (original_score - best_score_for_n)

    # Add skipped
    if TARGET_PUZZLES:
        skipped = df[~df['puzzle_n'].isin(TARGET_PUZZLES)]
        if not skipped.empty:
            final_dfs.append(skipped)
            
    final_df = pd.concat(final_dfs).sort_values('id')
    
    for col in ['x', 'y', 'deg']:
        final_df[col] = final_df[col].apply(lambda x: f"s{x:.20f}")
        
    final_df = final_df[['id', 'x', 'y', 'deg']]
    final_df.to_csv(output_file, index=False)
    
    print("\n" + "="*30)
    print(f"Best score for puzzle {n}: {best_score_for_n:.10f}")
    print(f"Total Improvement: {total_imp_global:.10f}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()