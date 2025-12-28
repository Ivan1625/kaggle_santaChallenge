"""
Christmas Tree Layout Optimization (Kaggle Version - Debug Optimized)
"""
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import time
import multiprocessing
import math
import random
import os
import glob
from traceback import format_exc
import sys

# --- Configuration ---
getcontext().prec = 50 
scale_factor = Decimal('1e18') 

# PATH CONFIGURATION
SEARCH_DIR = "/kaggle/input/submission-86"
OUTPUT_CSV = "/kaggle/working/submission.csv"

# TIME & ITERATION CONFIGURATION
MAX_ITER_PER_GROUP = 2500000  
T_START = 3  
T_END = 0.001  
LOCAL_TIME_LIMIT_HOURS = 11.0  # Increased to 11 hours
SAVE_EVERY_N_GROUPS = 50  
CHUNKSIZE = 1 

# --- Core Class ---
class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self.polygon = self._create_polygon()

    def _create_polygon(self):
        trunk_w = Decimal('0.15'); trunk_h = Decimal('0.2')
        base_w = Decimal('0.7'); base_y = Decimal('0.0')
        mid_w = Decimal('0.4'); tier_2_y = Decimal('0.25')
        top_w = Decimal('0.25'); tier_1_y = Decimal('0.5')
        tip_y = Decimal('0.8'); trunk_bottom_y = -trunk_h

        initial_polygon = Polygon([
            (Decimal('0.0') * scale_factor, tip_y * scale_factor),
            (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        return affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )

# --- Helper Functions ---
def get_tree_list_side_length_fast(polygons) -> float:
    if not polygons: return 0.0
    minx, miny, maxx, maxy = polygons[0].bounds
    for p in polygons[1:]:
        b = p.bounds
        if b[0] < minx: minx = b[0]
        if b[1] < miny: miny = b[1]
        if b[2] > maxx: maxx = b[2]
        if b[3] > maxy: maxy = b[3]
    return max(maxx - minx, maxy - miny) / float(scale_factor)

def validate_no_overlaps(polygons):
    if not polygons: return True
    strtree = STRtree(polygons)
    for i, poly in enumerate(polygons):
        candidates = strtree.query(poly)
        for cand in candidates:
            if hasattr(cand, "geom_type"):
                other = cand
                if other is poly: continue
            else:
                j = int(cand)
                if j == i: continue
                other = polygons[j]
            if (not poly.disjoint(other)) and (not poly.touches(other)):
                return False
    return True

def find_and_load_csv(search_path):
    # Try to find CSV in the directory
    if os.path.isdir(search_path):
        candidates = glob.glob(os.path.join(search_path, "**/*.csv"), recursive=True)
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {search_path}")
        target = candidates[0]
    else:
        target = search_path

    print(f'DEBUG: Loading CSV file: {target}', flush=True)
    result = pd.read_csv(target, encoding='utf-8')
    
    # Cleaning
    for col in ['x', 'y', 'deg']:
        if col in result.columns and result[col].dtype == object:
            result[col] = result[col].astype(str).str.strip('s')
    
    try:
        result[['group_id', 'item_id']] = result['id'].str.split('_', n=2, expand=True)
    except Exception:
        pass # Fallback handled by groupby

    dict_of_tree_list = {}
    for group_id, group_data in result.groupby('group_id'):
        tree_list = [
            ChristmasTree(center_x=str(row.x), center_y=str(row.y), angle=str(row.deg))
            for row in group_data.itertuples(index=False)
        ]
        dict_of_tree_list[group_id] = tree_list
    
    print(f"DEBUG: CSV loaded - {len(dict_of_tree_list)} groups ready.", flush=True)
    return dict_of_tree_list

def save_dict_to_csv(dict_of_tree_list, output_path):
    data = []
    try:
        sorted_keys = sorted(dict_of_tree_list.keys(), key=lambda x: int(x))
    except:
        sorted_keys = sorted(dict_of_tree_list.keys())

    for group_id in sorted_keys:
        trees = dict_of_tree_list[group_id]
        for i, tree in enumerate(trees):
            data.append({
                'id': f"{group_id}_{i}",
                'x': f"s{tree.center_x}",
                'y': f"s{tree.center_y}",
                'deg': f"s{tree.angle}",
            })
    
    df = pd.DataFrame(data)[['id', 'x', 'y', 'deg']]
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"DEBUG: Checkpoint saved to {output_path}", flush=True)

# --- Optimization Task ---
def run_simulated_annealing(args):
    # Added try/except block to catch worker crashes
    try:
        group_id, initial_trees, max_iterations, t_start, t_end = args
        n_trees = len(initial_trees)
        
        # Determine params
        is_small_n = n_trees <= 100
        if is_small_n:
            effective_max_iter = max_iterations * 3
            effective_t_start = t_start * 2.0
            gravity_weight = 1e-4
        else:
            effective_max_iter = max_iterations
            effective_t_start = t_start
            gravity_weight = 1e-6

        # Initialize State
        state = []
        for t in initial_trees:
            cx_float = float(t.center_x) * float(scale_factor)
            cy_float = float(t.center_y) * float(scale_factor)
            state.append({
                'poly': t.polygon,
                'cx': cx_float,
                'cy': cy_float,
                'angle': float(t.angle),
            })

        current_polys = [s['poly'] for s in state]
        current_bounds = [p.bounds for p in current_polys]
        scale_f = float(scale_factor)
        inv_scale_f = 1.0 / scale_f
        inv_scale_f2 = 1.0 / (scale_f * scale_f)

        def _envelope_from_bounds(bounds_list):
            if not bounds_list: return (0.0, 0.0, 0.0, 0.0)
            minx, miny, maxx, maxy = bounds_list[0]
            for b in bounds_list[1:]:
                if b[0] < minx: minx = b[0]
                if b[1] < miny: miny = b[1]
                if b[2] > maxx: maxx = b[2]
                if b[3] > maxy: maxy = b[3]
            return (minx, miny, maxx, maxy)

        def _envelope_from_bounds_replace(bounds_list, replace_i, replace_bounds):
            if not bounds_list: return (0.0, 0.0, 0.0, 0.0)
            b0 = replace_bounds if replace_i == 0 else bounds_list[0]
            minx, miny, maxx, maxy = b0
            for i, b in enumerate(bounds_list[1:], start=1):
                if i == replace_i: b = replace_bounds
                if b[0] < minx: minx = b[0]
                if b[1] < miny: miny = b[1]
                if b[2] > maxx: maxx = b[2]
                if b[3] > maxy: maxy = b[3]
            return (minx, miny, maxx, maxy)

        def _side_len_from_env(env):
            minx, miny, maxx, maxy = env
            return max(maxx - minx, maxy - miny) * inv_scale_f

        env = _envelope_from_bounds(current_bounds)
        dist_sum = sum(s['cx'] * s['cx'] + s['cy'] * s['cy'] for s in state)

        def energy_from(env_local, dist_sum_local):
            side_len = _side_len_from_env(env_local)
            normalized_dist = (dist_sum_local * inv_scale_f2) / max(1, n_trees)
            return side_len + gravity_weight * normalized_dist, side_len

        current_energy, current_side_len = energy_from(env, dist_sum)
        best_state_params = [{'cx': s['cx'], 'cy': s['cy'], 'angle': s['angle']} for s in state]
        best_real_score = current_side_len

        T = effective_t_start
        cooling_rate = math.pow(t_end / effective_t_start, 1.0 / effective_max_iter)

        for i in range(effective_max_iter):
            progress = i / effective_max_iter
            if is_small_n:
                move_scale = max(0.001, 1.0 * (1 - progress))
                rotate_scale = max(0.0005, 1.5 * (1 - progress))
            else:
                move_scale = max(0.0005, 0.5 * (T / effective_t_start))
                rotate_scale = max(0.0005, 3.0 * (T / effective_t_start))

            idx = random.randint(0, n_trees - 1)
            target = state[idx]
            orig_poly = target['poly']
            orig_bounds = current_bounds[idx]
            orig_cx, orig_cy, orig_angle = target['cx'], target['cy'], target['angle']

            dx = (random.random() - 0.5) * scale_f * 0.1 * move_scale
            dy = (random.random() - 0.5) * scale_f * 0.1 * move_scale
            d_angle = (random.random() - 0.5) * rotate_scale

            rotated_poly = affinity.rotate(orig_poly, d_angle, origin=(orig_cx, orig_cy))
            new_poly = affinity.translate(rotated_poly, xoff=dx, yoff=dy)
            new_bounds = new_poly.bounds
            new_cx = orig_cx + dx
            new_cy = orig_cy + dy
            new_angle = orig_angle + d_angle

            # Collision Check
            collision = False
            for k in range(n_trees):
                if k == idx: continue
                ox1, oy1, ox2, oy2 = current_bounds[k]
                if new_bounds[0] > ox2 or new_bounds[2] < ox1 or new_bounds[1] > oy2 or new_bounds[3] < oy1:
                    continue
                if (not new_poly.disjoint(current_polys[k])) and (not new_poly.touches(current_polys[k])):
                    collision = True
                    break
            
            if collision:
                T *= cooling_rate
                continue

            old_d = orig_cx * orig_cx + orig_cy * orig_cy
            new_d = new_cx * new_cx + new_cy * new_cy
            cand_dist_sum = dist_sum - old_d + new_d

            env_minx, env_miny, env_maxx, env_maxy = env
            need_recompute = (
                (orig_bounds[0] == env_minx and new_bounds[0] > env_minx) or
                (orig_bounds[1] == env_miny and new_bounds[1] > env_miny) or
                (orig_bounds[2] == env_maxx and new_bounds[2] < env_maxx) or
                (orig_bounds[3] == env_maxy and new_bounds[3] < env_maxy)
            )
            if need_recompute:
                cand_env = _envelope_from_bounds_replace(current_bounds, idx, new_bounds)
            else:
                cand_env = (
                    min(env_minx, new_bounds[0]),
                    min(env_miny, new_bounds[1]),
                    max(env_maxx, new_bounds[2]),
                    max(env_maxy, new_bounds[3]),
                )

            new_energy, new_real_score = energy_from(cand_env, cand_dist_sum)
            delta = new_energy - current_energy

            accept = False
            if delta < 0:
                accept = True
            elif T > 1e-10:
                prob = math.exp(-delta * 1000 / T)
                accept = random.random() < prob

            if accept:
                current_polys[idx] = new_poly
                current_bounds[idx] = new_bounds
                target['poly'] = new_poly
                target['cx'] = new_cx
                target['cy'] = new_cy
                target['angle'] = new_angle
                current_energy = new_energy
                env = cand_env
                dist_sum = cand_dist_sum

                if new_real_score < best_real_score:
                    best_real_score = new_real_score
                    for k in range(n_trees):
                        best_state_params[k]['cx'] = state[k]['cx']
                        best_state_params[k]['cy'] = state[k]['cy']
                        best_state_params[k]['angle'] = state[k]['angle']

            T *= cooling_rate

        # Reconstruct
        final_trees = []
        final_polys_check = []
        for p in best_state_params:
            cx_dec = Decimal(p['cx']) / scale_factor
            cy_dec = Decimal(p['cy']) / scale_factor
            angle_dec = Decimal(p['angle'])
            new_t = ChristmasTree(str(cx_dec), str(cy_dec), str(angle_dec))
            final_trees.append(new_t)
            final_polys_check.append(new_t.polygon)

        if not validate_no_overlaps(final_polys_check):
            orig_score = get_tree_list_side_length_fast([t.polygon for t in initial_trees])
            return group_id, initial_trees, orig_score, None

        return group_id, final_trees, best_real_score, None

    except Exception as e:
        return args[0], [], 0, str(e)

# --- Main ---
def main():
    print("="*50, flush=True)
    print("Christmas Tree Layout Optimization (Kaggle)", flush=True)
    print("="*50, flush=True)
    
    time_limit_sec = LOCAL_TIME_LIMIT_HOURS * 3600
    
    try:
        dict_of_tree_list = find_and_load_csv(SEARCH_DIR)
        
        try:
            groups_to_optimize = sorted(dict_of_tree_list.keys(), key=lambda x: int(x), reverse=True)
        except:
             groups_to_optimize = sorted(dict_of_tree_list.keys(), reverse=True)

        tasks = []
        for gid in groups_to_optimize:
            tasks.append((gid, dict_of_tree_list[gid], MAX_ITER_PER_GROUP, T_START, T_END))
        
        num_processes = multiprocessing.cpu_count()
        print(f"\n[System] detected {num_processes} cores.", flush=True)
        print(f"[System] Starting pool with {num_processes} workers.", flush=True)
        print(f"[System] NOTE: 3M iterations is SLOW. First output may take 10-20 mins. Please wait.", flush=True)

        start_time = time.time()
        improved_count = 0
        total_tasks = len(tasks)
        finished_tasks = 0

        pool = multiprocessing.Pool(processes=num_processes)
        # Using 1 as chunksize to get updates as soon as ANY group finishes
        results_iter = pool.imap_unordered(run_simulated_annealing, tasks, chunksize=1)

        for result in results_iter:
            group_id, optimized_trees, score, error_msg = result
            finished_tasks += 1

            if error_msg:
                print(f"[ERROR] Group {group_id} crashed: {error_msg}", flush=True)
                continue

            orig_polys = [t.polygon for t in dict_of_tree_list[group_id]]
            orig_score = get_tree_list_side_length_fast(orig_polys)

            status_msg = ""
            if score < orig_score and (orig_score - score) > 1e-12:
                status_msg = f" -> Improved (-{(orig_score - score):.6f})"
                dict_of_tree_list[group_id] = optimized_trees
                improved_count += 1

            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit_sec:
                print(f"\n[TIMEOUT] Reached {elapsed_time/3600:.2f}h. Saving...", flush=True)
                pool.terminate()
                break

            if finished_tasks % SAVE_EVERY_N_GROUPS == 0:
                print(f"\n[Auto-save] {finished_tasks}/{total_tasks} done...", flush=True)
                save_dict_to_csv(dict_of_tree_list, OUTPUT_CSV)

            print(f"[{finished_tasks}/{total_tasks}] Grp {group_id}: {orig_score:.4f}->{score:.4f} {status_msg}", flush=True)

        pool.close()
        pool.join()
        print(f"Done! Runtime: {time.time()-start_time:.2f}s", flush=True)

    except Exception as e:
        print(f"Fatal Error: {str(e)}", flush=True)
        print(format_exc(), flush=True)
    finally:
        if 'dict_of_tree_list' in locals():
            save_dict_to_csv(dict_of_tree_list, OUTPUT_CSV)
        print("="*50, flush=True)

if __name__ == '__main__':
    main()
