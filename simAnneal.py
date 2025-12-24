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
    def cross(ax, ay, bx, by): return ax*by - ay*bx
    r_x = a2x - a1x; r_y = a2y - a1y
    s_x = b2x - b1x; s_y = b2y - b1y
    denom = cross(r_x, r_y, s_x, s_y)
    if np.abs(denom) < 1e-12: return False
    u_numer = cross(b1x - a1x, b1y - a1y, r_x, r_y)
    t_numer = cross(b1x - a1x, b1y - a1y, s_x, s_y)
    u = u_numer / denom
    t = t_numer / denom
    eps = 1e-9
    if (t > -eps) and (t < 1.0 + eps) and (u > -eps) and (u < 1.0 + eps):
        return True
    return False

@njit(cache=True)
def check_poly_overlap(p1x, p1y, p2x, p2y, safety_margin):
    minx1, miny1, maxx1, maxy1 = get_bounds(p1x, p1y)
    minx2, miny2, maxx2, maxy2 = get_bounds(p2x, p2y)
    if (minx1 > maxx2 + safety_margin or maxx1 < minx2 - safety_margin or 
        miny1 > maxy2 + safety_margin or maxy1 < miny2 - safety_margin):
        return False
    for i in range(NUM_VERTS - 1):
        if is_point_in_poly(p1x[i], p1y[i], p2x, p2y): return True
    for i in range(NUM_VERTS - 1):
        if is_point_in_poly(p2x[i], p2y[i], p1x, p1y): return True
    for i in range(NUM_VERTS - 1):
        for j in range(NUM_VERTS - 1):
            if segments_intersect(p1x[i], p1y[i], p1x[i+1], p1y[i+1],
                                  p2x[j], p2y[j], p2x[j+1], p2y[j+1]):
                return True
    return False

@njit(cache=True)
def get_system_score_raw(xs, ys, rots, n, temp_cache_x, temp_cache_y):
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
    cache_x = np.zeros((n, 16), dtype=np.float64)
    cache_y = np.zeros((n, 16), dtype=np.float64)
    for i in range(n):
        transform_poly(xs[i], ys[i], rots[i], cache_x[i], cache_y[i])
        
    current_area = get_system_score_raw(xs, ys, rots, n, cache_x, cache_y)
    best_area = current_area
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

    for k in range(iterations):
        # 1. HYDRAULIC PRESS (Squeeze)
        did_squeeze = False
        if k % squeeze_freq == 0:
            saved_xs = xs.copy()
            saved_ys = ys.copy()
            squeeze_valid = True
            
            # Pull towards center
            for i in range(n):
                xs[i] = cx + (xs[i] - cx) * squeeze_factor
                ys[i] = cy + (ys[i] - cy) * squeeze_factor
            
            # Check validity
            for i in range(n):
                transform_poly(xs[i], ys[i], rots[i], cache_x[i], cache_y[i])
            for i in range(n):
                for j in range(i+1, n):
                    if check_poly_overlap(cache_x[i], cache_y[i], cache_x[j], cache_y[j], safety_margin):
                        squeeze_valid = False
                        break
                if not squeeze_valid: break
            
            if squeeze_valid:
                did_squeeze = True
                current_area = get_system_score_raw(xs, ys, rots, n, cache_x, cache_y)
                if current_area < best_area:
                    best_area = current_area
                    best_xs = xs.copy()
                    best_ys = ys.copy()
                    best_rots = rots.copy()
            else:
                # Revert
                xs[:] = saved_xs[:]
                ys[:] = saved_ys[:]
                for i in range(n):
                    transform_poly(xs[i], ys[i], rots[i], cache_x[i], cache_y[i])

        # 2. PERTURBATION
        if not did_squeeze:
            idx = np.random.randint(0, n)
            old_x, old_y, old_rot = xs[idx], ys[idx], rots[idx]
            
            rnd = np.random.random()
            mag = max(0.00000001, temp * 0.05) 
            
            if rnd < 0.5: 
                xs[idx] += np.random.normal(0, mag)
                ys[idx] += np.random.normal(0, mag)
            elif rnd < 0.9: 
                rots[idx] += np.random.normal(0, mag * 50.0) 
            else: 
                # Larger jumps to fix bad packing
                if temp > 1.0:
                    xs[idx] += np.random.normal(0, 0.5)
                    ys[idx] += np.random.normal(0, 0.5)
                    rots[idx] = np.random.uniform(0, 360)

            tx = np.zeros(16, dtype=np.float64)
            ty = np.zeros(16, dtype=np.float64)
            transform_poly(xs[idx], ys[idx], rots[idx], tx, ty)
            
            valid = True
            for j in range(n):
                if idx == j: continue
                if check_poly_overlap(tx, ty, cache_x[j], cache_y[j], safety_margin):
                    valid = False
                    break
            
            if valid:
                cache_x[idx] = tx
                cache_y[idx] = ty
                new_area = get_system_score_raw(xs, ys, rots, n, cache_x, cache_y)
                delta = new_area - current_area
                
                if delta < 0 or np.random.random() < np.exp(-delta / temp):
                    current_area = new_area
                    if current_area < best_area:
                        best_area = current_area
                        best_xs = xs.copy()
                        best_ys = ys.copy()
                        best_rots = rots.copy()
                else:
                    xs[idx] = old_x
                    ys[idx] = old_y
                    rots[idx] = old_rot
                    transform_poly(old_x, old_y, old_rot, cache_x[idx], cache_y[idx])
            else:
                xs[idx] = old_x
                ys[idx] = old_y
                rots[idx] = old_rot

        temp *= cooling_rate
        if temp < 1e-12: temp = 1e-12

    return best_xs, best_ys, best_rots, best_area

# ==========================================
# 4. INITIALIZATION & WRAPPER
# ==========================================

def get_base_shapely_poly():
    X = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, 
         -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125, 0.0]
    Y = [0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, 
         -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.8]
    return Polygon(list(zip(X, Y)))

BASE_SHAPELY = get_base_shapely_poly()

def verify_validity(xs, ys, rots):
    """Shapely verification."""
    polys = []
    n = len(xs)
    for i in range(n):
        p = affinity.rotate(BASE_SHAPELY, rots[i], origin=(0,0))
        p = affinity.translate(p, xs[i], ys[i])
        polys.append(p)
    
    for i in range(n):
        p1 = polys[i].buffer(-1e-9) 
        for j in range(i+1, n):
            if p1.intersects(polys[j]):
                return False
    return True

def generate_grid_initialization(n):
    """
    Generates a safe, dispersed grid layout.
    """
    xs = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    rots = np.zeros(n, dtype=np.float64)
    ids = [f"{n}_{i}" for i in range(n)]
    
    # Calculate grid size
    grid_side = math.ceil(math.sqrt(n))
    spacing = 2.0  # Safe distance (Tree is ~1.0 high, 0.7 wide)
    
    for i in range(n):
        row = i // grid_side
        col = i % grid_side
        
        # Center the grid around 0,0 for easier squeezing
        xs[i] = (col - grid_side/2) * spacing
        ys[i] = (row - grid_side/2) * spacing
        
        # Random initial rotation
        rots[i] = random.uniform(0, 360)
        
    return pd.DataFrame({'id': ids, 'x': xs, 'y': ys, 'deg': rots})

def optimize_from_scratch(n, params):
    # 1. Generate Start State
    df_init = generate_grid_initialization(n)
    
    xs = np.ascontiguousarray(df_init['x'].values, dtype=np.float64)
    ys = np.ascontiguousarray(df_init['y'].values, dtype=np.float64)
    rots = np.ascontiguousarray(df_init['deg'].values, dtype=np.float64)
    
    # JIT Warmup
    if not hasattr(optimize_from_scratch, "compiled"):
        run_strict_sa(xs.copy(), ys.copy(), rots.copy(), n, 10, 0.1, 0.9, 1.0, 100, 1e-7)
        optimize_from_scratch.compiled = True
        
    best_xs, best_ys, best_rots, best_raw_area = run_strict_sa(
        xs, ys, rots, n, 
        iterations=params['ITERATIONS'], 
        start_temp=params['START_TEMP'],
        cooling_rate=params['COOLING_RATE'], 
        squeeze_factor=params['SQUEEZE_FACTOR'],
        squeeze_freq=params['SQUEEZE_FREQ'],
        safety_margin=params['SAFETY_MARGIN']
    )
    
    # Final Check
    if not verify_validity(best_xs, best_ys, best_rots):
        return None, float('inf')
    
    res_data = []
    for i in range(n):
        res_data.append({
            'id': df_init.iloc[i]['id'],
            'x': best_xs[i],
            'y': best_ys[i],
            'deg': best_rots[i]
        })
        
    final_score = best_raw_area / n
    return pd.DataFrame(res_data), final_score

def main():
    # =====================
    # CONFIGURATION
    # =====================
    output_file = 'C:/Users/user/Downloads/submission_from_scratch.csv'
    
    # WHICH N TO SOLVE?
    TARGET_NS = [16] 
    
    # HOW MANY RESTARTS?
    LOOPS_PER_N = 10
    
    PARAMS = {
        'ITERATIONS': 500_000,    # Increase because we start far apart
        'START_TEMP': 5.0,       # High temp to bring them together fast
        'COOLING_RATE': 0.99998,  # Slow cool for packing
        'SQUEEZE_FACTOR': 0.999,  # Slightly more aggressive squeeze initally
        'SQUEEZE_FREQ': 50,       # Frequent squeezing
        'SAFETY_MARGIN': 1e-20
    }
    # =====================
    
    final_dfs = []
    
    print(f"Generating and Optimizing for N = {TARGET_NS}")
    print(f"{'N':<5} | {'Loop':<5} | {'Best Score':<15} | {'Status'}")
    print("-" * 50)
    
    for n in TARGET_NS:
        
        best_df_for_n = None
        best_score_for_n = float('inf')
        
        for i in range(LOOPS_PER_N):
            # Run optimization from fresh random grid
            opt_df, new_score = optimize_from_scratch(n, PARAMS)
            
            status = "Discarded"
            if opt_df is not None:
                if new_score < best_score_for_n:
                    best_score_for_n = new_score
                    best_df_for_n = opt_df
                    status = "NEW BEST"
                else:
                    status = "Valid (Worse)"
            else:
                status = "Invalid"
                
            print(f"{n:<5} | {i+1}/{LOOPS_PER_N} | {best_score_for_n:.8f}    | {status}")
            
        if best_df_for_n is not None:
            final_dfs.append(best_df_for_n)
            print(f"--> Finished N={n}. Final Score: {best_score_for_n:.8f}\n")
        else:
            print(f"--> Failed to find valid solution for N={n} (Check params)\n")

    # Save
    if final_dfs:
        final_df = pd.concat(final_dfs).sort_values('id')
        
        # Format
        for col in ['x', 'y', 'deg']:
            final_df[col] = final_df[col].apply(lambda x: f"s{x:.20f}")
            
        final_df = final_df[['id', 'x', 'y', 'deg']]
        final_df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")
    else:
        print("No valid solutions found.")

if __name__ == "__main__":
    main()