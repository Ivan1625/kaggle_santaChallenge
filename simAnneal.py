"""
Christmas Tree Layout Optimization (Local Run Version)
Core Functions:
    1. Read Christmas tree parameters (group ID, coordinates, rotation angle) from local CSV
    2. Run simulated annealing algorithm with multiprocessing to optimize layout per group
    3. Optimization Goal: Minimize bounding box side length + No area overlap (edge/point contact only)
    4. Auto-save progress periodically, support manual interruption with progress saving
Dependencies: pip install pandas shapely
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
from traceback import format_exc


#scaled to avoid floating point errors
getcontext().prec = 50 
scale_factor = Decimal('1e18') 

# --- Configuration Parameters ---
# input output csv; max iter per grp; temp start/end; time limit hrs; save freq; chunk size
INPUT_CSV = r"C:\Users\user\OneDrive\Desktop\Python Folder\kaggle\Scores\sample_submission.csv"
OUTPUT_CSV = r"C:\Users\user\OneDrive\Desktop\Python Folder\kaggle\submission_sa (1).csv"  
MAX_ITER_PER_GROUP = 500000  
T_START = 3  
T_END = 0.001  
LOCAL_TIME_LIMIT_HOURS = 0  
SAVE_EVERY_N_GROUPS = 50  
CHUNKSIZE = 1 

# --- Core Class: Christmas Tree Geometric Model ---
class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        """
        Initialize Christmas Tree
        :param center_x: Center x coordinate (string/number)
        :param center_y: Center y coordinate (string/number)
        :param angle: Rotation angle (degrees, string/number)
        """
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self.polygon = self._create_polygon()

    def _create_polygon(self):
        """Build Christmas tree polygon (trunk + 3 tiers), apply rotation + translation"""
        # Christmas tree dimension parameters
        trunk_w = Decimal('0.15'); trunk_h = Decimal('0.2')
        base_w = Decimal('0.7'); base_y = Decimal('0.0')
        mid_w = Decimal('0.4'); tier_2_y = Decimal('0.25')
        top_w = Decimal('0.25'); tier_1_y = Decimal('0.5')
        tip_y = Decimal('0.8'); trunk_bottom_y = -trunk_h

        # Build initial polygon vertices (scale to avoid floating point errors)
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
        # Rotate + Translate
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        return affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )

    def clone(self) -> "ChristmasTree":
        """Clone ChristmasTree object (avoid re-calculating polygon)"""
        new_tree = ChristmasTree.__new__(ChristmasTree)
        new_tree.center_x = self.center_x
        new_tree.center_y = self.center_y
        new_tree.angle = self.angle
        new_tree.polygon = self.polygon
        return new_tree

# --- Helper Functions ---
def get_tree_list_side_length_fast(polygons) -> float:
    """Fast calculate max side length of polygon group's bounding box (restore scale factor)"""
    if not polygons:
        return 0.0
    minx, miny, maxx, maxy = polygons[0].bounds
    for p in polygons[1:]:
        b = p.bounds
        if b[0] < minx: minx = b[0]
        if b[1] < miny: miny = b[1]
        if b[2] > maxx: maxx = b[2]
        if b[3] > maxy: maxy = b[3]
    return max(maxx - minx, maxy - miny) / float(scale_factor)

def validate_no_overlaps(polygons):
    """Use spatial index to detect polygon area overlap (edge/point contact only allowed)"""
    if not polygons:
        return True

    strtree = STRtree(polygons)
    for i, poly in enumerate(polygons):
        candidates = strtree.query(poly)
        for cand in candidates:
            # Compatible with Shapely 1.8/2.x
            if hasattr(cand, "geom_type"):
                other = cand
                if other is poly:
                    continue
            else:
                j = int(cand)
                if j == i:
                    continue
                other = polygons[j]
            # Not disjoint and not only touching = area overlap
            if (not poly.disjoint(other)) and (not poly.touches(other)):
                return False
    return True

def parse_csv(csv_path):
    """Parse local CSV file, generate tree list grouped by group ID"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file does not exist: {csv_path}")
    
    print(f'Loading CSV file: {csv_path}')
    # Read CSV with specified encoding (avoid local Chinese garbled characters)
    result = pd.read_csv(csv_path, encoding='utf-8')
    
    # Verify required columns exist
    required_cols = ['id', 'x', 'y', 'deg']
    missing_cols = [col for col in required_cols if col not in result.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Clean x/y/deg columns (remove possible 's' prefix)
    for col in ['x', 'y', 'deg']:
        if result[col].dtype == object:
            result[col] = result[col].astype(str).str.strip('s')
    
    # Split id column into group_id and item_id
    result[['group_id', 'item_id']] = result['id'].str.split('_', n=2, expand=True)
    if result['group_id'].isnull().any():
        raise ValueError("Invalid ID column format - must be 'group_id_item_id' (e.g., 1_0)")
    
    # Generate tree list by group
    dict_of_tree_list = {}
    for group_id, group_data in result.groupby('group_id'):
        tree_list = [
            ChristmasTree(center_x=str(row.x), center_y=str(row.y), angle=str(row.deg))
            for row in group_data.itertuples(index=False)
        ]
        dict_of_tree_list[group_id] = tree_list
    
    print(f"CSV loaded successfully - {len(dict_of_tree_list)} tree groups found")
    return dict_of_tree_list

def save_dict_to_csv(dict_of_tree_list, output_path):
    """Save optimization results to local CSV file"""
    print(f"Saving results to: {output_path}")
    # Create output directory if not exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Organize data
    data = []
    sorted_keys = sorted(dict_of_tree_list.keys(), key=lambda x: int(x))
    for group_id in sorted_keys:
        trees = dict_of_tree_list[group_id]
        for i, tree in enumerate(trees):
            data.append({
                'id': f"{group_id}_{i}",
                'x': f"s{tree.center_x}",
                'y': f"s{tree.center_y}",
                'deg': f"s{tree.angle}",
            })
    
    # Save CSV (specify encoding)
    df = pd.DataFrame(data)[['id', 'x', 'y', 'deg']]
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Save completed! File path: {output_path}")

# --- Core Simulated Annealing Function ---
def run_simulated_annealing(args):
    """Simulated annealing optimization for single tree group (multiprocessing task)"""
    group_id, initial_trees, max_iterations, t_start, t_end = args
    n_trees = len(initial_trees)

    # Adjust parameters based on number of trees
    is_small_n = n_trees <= 100
    if is_small_n:
        effective_max_iter = max_iterations * 3
        effective_t_start = t_start * 2.0
        gravity_weight = 1e-4
    else:
        effective_max_iter = max_iterations
        effective_t_start = t_start
        gravity_weight = 1e-6

    # Initialize state (Decimal to float for faster calculation)
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

    # Helper function: Calculate overall bounding box
    def _envelope_from_bounds(bounds_list):
        if not bounds_list:
            return (0.0, 0.0, 0.0, 0.0)
        minx, miny, maxx, maxy = bounds_list[0]
        for b in bounds_list[1:]:
            if b[0] < minx: minx = b[0]
            if b[1] < miny: miny = b[1]
            if b[2] > maxx: maxx = b[2]
            if b[3] > maxy: maxy = b[3]
        return (minx, miny, maxx, maxy)

    def _envelope_from_bounds_replace(bounds_list, replace_i: int, replace_bounds):
        """Incremental update of bounding box"""
        if not bounds_list:
            return (0.0, 0.0, 0.0, 0.0)
        b0 = replace_bounds if replace_i == 0 else bounds_list[0]
        minx, miny, maxx, maxy = b0
        for i, b in enumerate(bounds_list[1:], start=1):
            if i == replace_i:
                b = replace_bounds
            if b[0] < minx: minx = b[0]
            if b[1] < miny: miny = b[1]
            if b[2] > maxx: maxx = b[2]
            if b[3] > maxy: maxy = b[3]
        return (minx, miny, maxx, maxy)

    def _side_len_from_env(env):
        minx, miny, maxx, maxy = env
        return max(maxx - minx, maxy - miny) * inv_scale_f

    # Initialize bounding box and distance sum
    env = _envelope_from_bounds(current_bounds)
    dist_sum = 0.0
    for s in state:
        dist_sum += s['cx'] * s['cx'] + s['cy'] * s['cy']

    # Energy function: Bounding box side length + gravity penalty (avoid trees moving away from center)
    def energy_from(env_local, dist_sum_local):
        side_len = _side_len_from_env(env_local)
        normalized_dist = (dist_sum_local * inv_scale_f2) / max(1, n_trees)
        return side_len + gravity_weight * normalized_dist, side_len

    current_energy, current_side_len = energy_from(env, dist_sum)
    best_state_params = [{'cx': s['cx'], 'cy': s['cy'], 'angle': s['angle']} for s in state]
    best_real_score = current_side_len

    # Simulated annealing cooling configuration
    T = effective_t_start
    cooling_rate = math.pow(t_end / effective_t_start, 1.0 / effective_max_iter)

    # Core iteration
    for i in range(effective_max_iter):
        progress = i / effective_max_iter
        # Dynamically adjust perturbation scale
        if is_small_n:
            move_scale = max(0.001, 1.0 * (1 - progress))
            rotate_scale = max(0.0005, 1.5 * (1 - progress))
        else:
            move_scale = max(0.0005, 0.5 * (T / effective_t_start))
            rotate_scale = max(0.0005, 3.0 * (T / effective_t_start))

        # Randomly select one tree for perturbation
        idx = random.randint(0, n_trees - 1)
        target = state[idx]
        orig_poly = target['poly']
        orig_bounds = current_bounds[idx]
        orig_cx, orig_cy, orig_angle = target['cx'], target['cy'], target['angle']

        # Generate random perturbation
        dx = (random.random() - 0.5) * scale_f * 0.1 * move_scale
        dy = (random.random() - 0.5) * scale_f * 0.1 * move_scale
        d_angle = (random.random() - 0.5) * rotate_scale

        # Apply rotation and translation
        rotated_poly = affinity.rotate(orig_poly, d_angle, origin=(orig_cx, orig_cy))
        new_poly = affinity.translate(rotated_poly, xoff=dx, yoff=dy)
        new_bounds = new_poly.bounds
        new_cx = orig_cx + dx
        new_cy = orig_cy + dy
        new_angle = orig_angle + d_angle

        # Collision detection
        collision = False
        for k in range(n_trees):
            if k == idx:
                continue
            ox1, oy1, ox2, oy2 = current_bounds[k]
            if new_bounds[0] > ox2 or new_bounds[2] < ox1 or new_bounds[1] > oy2 or new_bounds[3] < oy1:
                continue
            other = current_polys[k]
            if (not new_poly.disjoint(other)) and (not new_poly.touches(other)):
                collision = True
                break
        if collision:
            T *= cooling_rate
            continue

        # Calculate new energy
        old_d = orig_cx * orig_cx + orig_cy * orig_cy
        new_d = new_cx * new_cx + new_cy * new_cy
        cand_dist_sum = dist_sum - old_d + new_d

        # Incremental update of bounding box
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

        # Metropolis criterion to accept new state
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 1e-10:
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

    # Generate final results
    final_trees = []
    final_polys_check = []
    for p in best_state_params:
        cx_dec = Decimal(p['cx']) / scale_factor
        cy_dec = Decimal(p['cy']) / scale_factor
        angle_dec = Decimal(p['angle'])
        new_t = ChristmasTree(str(cx_dec), str(cy_dec), str(angle_dec))
        final_trees.append(new_t)
        final_polys_check.append(new_t.polygon)

    # Final validation: return original trees if overlap exists
    if not validate_no_overlaps(final_polys_check):
        orig_score = get_tree_list_side_length_fast([t.polygon for t in initial_trees])
        return group_id, initial_trees, orig_score

    return group_id, final_trees, best_real_score

# --- Main Function (Local Run Entry) ---
def main():
    print("="*50)
    print("Christmas Tree Layout Optimization (Local Version)")
    print("="*50)
    
    # Convert time limit to seconds
    time_limit_sec = LOCAL_TIME_LIMIT_HOURS * 3600 if LOCAL_TIME_LIMIT_HOURS > 0 else float('inf')
    
    try:
        # 1. Read input CSV
        dict_of_tree_list = parse_csv(INPUT_CSV)
        
        # 2. Generate optimization tasks
        groups_to_optimize = sorted(dict_of_tree_list.keys(), key=lambda x: int(x), reverse=True)
        tasks = []
        for gid in groups_to_optimize:
            tasks.append((gid, dict_of_tree_list[gid], MAX_ITER_PER_GROUP, T_START, T_END))
        
        # 3. Configure multiprocessing
        num_processes = multiprocessing.cpu_count() - 1  # Leave 1 core for system
        num_processes = max(1, num_processes)  # At least 1 process
        print(f"\nOptimization Configuration:")
        print(f"- Number of tree groups to optimize: {len(tasks)}")
        print(f"- Number of processes enabled: {num_processes}")
        print(f"- Max iterations per group: {MAX_ITER_PER_GROUP}")
        print(f"- Runtime limit: {LOCAL_TIME_LIMIT_HOURS} hours (0 = no limit)")
        print(f"- Auto-save interval: Every {SAVE_EVERY_N_GROUPS} groups")
        print(f"- Press Ctrl+C to interrupt manually and save progress\n")

        # 4. Initialize monitoring variables
        start_time = time.time()
        improved_count = 0
        total_tasks = len(tasks)
        finished_tasks = 0

        # 5. Start multiprocessing pool
        pool = multiprocessing.Pool(processes=num_processes)
        results_iter = pool.imap_unordered(run_simulated_annealing, tasks, chunksize=CHUNKSIZE)

        # 6. Process optimization results
        for result in results_iter:
            group_id, optimized_trees, score = result
            finished_tasks += 1

            # Calculate original score
            orig_polys = [t.polygon for t in dict_of_tree_list[group_id]]
            orig_score = get_tree_list_side_length_fast(orig_polys)

            # Check if optimization succeeded
            status_msg = ""
            if score < orig_score and (orig_score - score) > 1e-12:
                status_msg = f" -> Optimization succeeded! (-{(orig_score - score):.6f})"
                dict_of_tree_list[group_id] = optimized_trees
                improved_count += 1

            # Check time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit_sec:
                print(f"\n[WARNING] Time limit reached ({elapsed_time/3600:.2f} hours), stopping optimization and saving...")
                pool.terminate()
                break

            # Auto-save progress
            if finished_tasks % SAVE_EVERY_N_GROUPS == 0:
                print(f"\n[Auto-save] Completed {finished_tasks}/{total_tasks} groups...")
                save_dict_to_csv(dict_of_tree_list, OUTPUT_CSV)

            # Print progress
            print(f"[{finished_tasks}/{total_tasks}] Group {group_id}: {orig_score:.5f} -> {score:.5f} {status_msg}")

        # All tasks completed normally
        pool.close()
        pool.join()
        print(f"All optimization tasks completed! Total runtime: {time.time()-start_time:.2f} seconds")

    except KeyboardInterrupt:
        # Manual interruption (Ctrl+C)
        print("Manual interruption detected (Ctrl+C), saving current progress...")
        if 'pool' in locals():
            pool.terminate()
            pool.join()
    except Exception as e:
        # Other errors
        print(f"Runtime error: {type(e).__name__} - {str(e)}")
        print(f"Detailed error information:\n{format_exc()}")
    finally:
        # Final save results
        if 'dict_of_tree_list' in locals():
            print(f"Saving final results...")
            save_dict_to_csv(dict_of_tree_list, OUTPUT_CSV)
            print(f"Optimization Statistics:")
            print(f"- Total groups: {len(dict_of_tree_list)}")
            print(f"- Successfully optimized groups: {improved_count}")
        print("\nProgram finished!")
        print("="*50)

if __name__ == '__main__':
    # Windows multiprocessing compatibility
    multiprocessing.freeze_support()
    main()

# Run Instructions:
# 1. Modify the "Local Run Parameter Configuration" section with your actual paths and parameters
# 2. Ensure input CSV format is correct (contains id/x/y/deg columns, id format: group_id_item_id)
# 3. Install dependencies: pip install pandas shapely
# 4. Run the script directly
