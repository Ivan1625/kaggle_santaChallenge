import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
from tqdm import tqdm

# ==========================================
# 1. GEOMETRY ENGINE
# ==========================================
def get_santa_tree_polygon():
    # Vertices from 'Getting Started' notebook
    X = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, 
         -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
    Y = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, 
         -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]
    vertices = list(zip(X, Y))
    return Polygon(vertices)

base_poly_template = get_santa_tree_polygon()

class PackingState:
    def __init__(self, items):
        self.items = items
        self.polygons = self._generate_polygons()
        self.bounds = self._calc_bounds()
        self.overlap_area = 0.0 # Calculated on demand
        self.square_size = self._calc_square_size()

    def _generate_polygons(self):
        polys = []
        for item in self.items:
            p = rotate(base_poly_template, item['rot'], origin=(0,0))
            p = translate(p, item['x'], item['y'])
            polys.append(p)
        return polys
    
    def _calc_bounds(self):
        min_x = min(p.bounds[0] for p in self.polygons)
        min_y = min(p.bounds[1] for p in self.polygons)
        max_x = max(p.bounds[2] for p in self.polygons)
        max_y = max(p.bounds[3] for p in self.polygons)
        return min_x, min_y, max_x, max_y

    def _calc_square_size(self):
        min_x, min_y, max_x, max_y = self.bounds
        return max(max_x - min_x, max_y - min_y)

    def calculate_total_overlap(self):
        """Expensive check: sums up the area of all overlaps."""
        total_overlap = 0.0
        # Check every pair
        n = len(self.polygons)
        for i in range(n):
            p1 = self.polygons[i]
            # Fast bound check optimization
            b1 = p1.bounds
            for j in range(i + 1, n):
                p2 = self.polygons[j]
                b2 = p2.bounds
                if (b1[2] < b2[0] or b1[0] > b2[2] or 
                    b1[3] < b2[1] or b1[1] > b2[3]):
                    continue
                    
                if p1.intersects(p2):
                    inter = p1.intersection(p2)
                    total_overlap += inter.area
        return total_overlap

# ==========================================
# 2. THE LOGIC: COMPRESS -> RESOLVE
# ==========================================

def compress_state(state, scale_factor=0.99):
    """
    Applies the 'Press': Moves everything towards the center.
    This WILL create overlaps.
    """
    min_x, min_y, max_x, max_y = state.bounds
    cx, cy = (min_x + max_x)/2, (min_y + max_y)/2
    
    new_items = []
    for item in state.items:
        # Vector to center
        dx = item['x'] - cx
        dy = item['y'] - cy
        
        # Pull in
        new_x = cx + dx * scale_factor
        new_y = cy + dy * scale_factor
        
        new_items.append({
            'id': item['id'], 
            'x': new_x, 
            'y': new_y, 
            'rot': item['rot']
        })
    return PackingState(new_items)

def resolve_collisions(compressed_state, max_steps=5000):
    """
    The 'Adjust' phase: Tries to fix the overlaps created by compression.
    Uses Simulated Annealing with a cost function dominated by Overlap Area.
    """
    current_state = copy.deepcopy(compressed_state)
    current_overlap = current_state.calculate_total_overlap()
    
    # If no overlap happened after compression (rare!), we are good.
    if current_overlap < 1e-9:
        return True, current_state
    
    temp = 1.0
    cooling = 0.99
    
    # We want to minimize overlap to 0
    
    for i in range(max_steps):
        if current_overlap < 1e-9:
            return True, current_state # Solved!
        
        # Pick random item
        idx = random.randint(0, len(current_state.items)-1)
        item_old = current_state.items[idx]
        item_new = item_old.copy()
        
        # MUTATION: Wiggle to find empty space
        # As temp decreases, wiggles get smaller
        mag = max(0.001, temp * 0.1)
        r = random.random()
        
        if r < 0.5:
            # Translation nudge
            item_new['x'] += random.gauss(0, mag)
            item_new['y'] += random.gauss(0, mag)
        elif r < 0.8:
            # Rotation wiggle
            item_new['rot'] = (item_new['rot'] + random.gauss(0, mag * 50)) % 360
        else:
            # Wild rotation (spin) - helps unlock jammed pieces
            item_new['rot'] = random.uniform(0, 360)
            
        # Update Geometry
        p_new = rotate(base_poly_template, item_new['rot'], origin=(0,0))
        p_new = translate(p_new, item_new['x'], item_new['y'])
        
        # Optimization: Calculate overlap change ONLY for this piece
        # Re-calculating all pairs is too slow.
        # Approximation: Just check if this piece's overlap reduced?
        # For correctness in this script, we'll do a partial recalculation logic:
        # (Old Total) - (Old Piece Contribution) + (New Piece Contribution)
        
        def get_piece_overlap(poly, all_polys, index):
            area = 0.0
            for k, p in enumerate(all_polys):
                if k == index: continue
                # Bound check
                if not poly.bounds[2] < p.bounds[0] and not poly.bounds[0] > p.bounds[2] and \
                   not poly.bounds[3] < p.bounds[1] and not poly.bounds[1] > p.bounds[3]:
                    if poly.intersects(p):
                        area += poly.intersection(p).area
            return area

        old_poly = current_state.polygons[idx]
        overlap_contribution_old = get_piece_overlap(old_poly, current_state.polygons, idx)
        
        overlap_contribution_new = get_piece_overlap(p_new, current_state.polygons, idx)
        
        delta_overlap = overlap_contribution_new - overlap_contribution_old
        
        # Also consider bounding box penalty slightly to prevent exploding outward
        # But primarily we care about overlap
        
        if delta_overlap < 0 or random.random() < math.exp(-delta_overlap * 100 / temp):
            # Accept
            current_state.items[idx] = item_new
            current_state.polygons[idx] = p_new
            current_overlap += delta_overlap
            
            # Correction for float drift
            if i % 100 == 0:
                current_overlap = current_state.calculate_total_overlap()
        
        temp *= cooling
        
    # Final check
    final_overlap = current_state.calculate_total_overlap()
    if final_overlap < 1e-9:
        return True, current_state
    
    return False, current_state # Failed to resolve

# ==========================================
# 3. MAIN "HYDRAULIC PRESS" LOOP
# ==========================================

def run_hydraulic_press(initial_state, cycles=100):
    """
    1. Compress.
    2. Try to resolve collisions.
    3. If resolved: Keep new state, Compress again.
    4. If failed: Revert, shake (high temp perturbation), try again.
    """
    valid_state = copy.deepcopy(initial_state)
    
    print(f"Initial Valid Side: {valid_state.square_size:.6f}")
    
    pbar = tqdm(range(cycles), desc="Pressing")
    
    for i in pbar:
        # 1. Attempt Compression (0.99x)
        # We start with aggressive compression, then slow down
        ratio = 0.98 if i < 10 else 0.995 
        compressed_candidate = compress_state(valid_state, scale_factor=ratio)
        
        # 2. Adjust (Resolve Collisions)
        success, resolved_state = resolve_collisions(compressed_candidate, max_steps=4000)
        
        if success:
            # We successfully shrank it and fixed the overlaps!
            valid_state = resolved_state
            
            # Optional: Center the packing to prevent drift
            min_x, min_y, max_x, max_y = valid_state.bounds
            cx, cy = (min_x+max_x)/2, (min_y+max_y)/2
            for item in valid_state.items:
                item['x'] -= cx
                item['y'] -= cy
            valid_state = PackingState(valid_state.items) # Re-init to update polys
            
        else:
            # The compression created a jam we couldn't fix.
            # Strategy: Don't compress this time. Just "Adjust" the VALID state
            # with high heat to rearrange the configuration for the next press.
            _, shuffled_state = resolve_collisions(valid_state, max_steps=2000) 
            # We don't check success here, we just accept the shuffling if it's valid
            if shuffled_state.calculate_total_overlap() < 1e-9:
                 valid_state = shuffled_state
        
        pbar.set_postfix({'Side': f"{valid_state.square_size:.6f}"})
        
    return valid_state

# ==========================================
# UTILS & MAIN
# ==========================================

def parse_input(data_str):
    items = []
    for line in data_str.strip().split('\n'):
        parts = line.split()
        idx = int(parts[0].split('_')[1])
        x = float(parts[1].replace('s', ''))
        y = float(parts[2].replace('s', ''))
        rot = float(parts[3].replace('s', ''))
        items.append({'id': idx, 'x': x, 'y': y, 'rot': rot})
    return PackingState(items)

def plot_final(state):
    min_x, min_y, max_x, max_y = state.bounds
    w, h = max_x - min_x, max_y - min_y
    side = max(w, h)
    
    fig, ax = plt.subplots(figsize=(10,10))
    for p in state.polygons:
        x, y = p.exterior.xy
        ax.fill(x, y, fc='forestgreen', ec='black', alpha=0.9, lw=0.5)
        
    rect = Rectangle((min_x, min_y), side, side, ec='red', fc='none', lw=2, label=f'Square: {side:.4f}')
    ax.add_patch(rect)
    ax.set_aspect('equal')
    ax.legend()
    plt.title(f"Hydraulic Press Result\nSquare Area: {side**2:.6f}")
    plt.show()

raw_input_data = """
016_0	s1.0996197078470942	s1.5593257672032839	s66.18900069193194
016_1	s0.21062450840936794	s1.602815894431431	s246.18888953083774
016_2	s0.21087355342791167	s0.391729707110211	s247.30818735449884
016_3	s2.233530680360916	s1.568751976668956	s66.18772508603041
016_4	s0.21062300973337472	s1.0158951659468354	s246.18702409799448
016_5	s1.1065238895395129	s2.1335146201510655	s66.22766147768715
016_6	s2.233531052390252	s0.981747844092935	s66.18596868937578
016_7	s1.343160299091237	s1.615813365425839	s246.18777145113472
016_8	s1.101014717672186	s0.9687650304129949	s66.18701908534007
016_9	s1.3534024539844383	s2.1939874250943436	s247.37048231476626
016_10	s1.344555335037425	s1.025243374137446	s246.18596795487724
016_11	s0.2110099398166373	s2.193055295579359	s246.2279908541534
016_12	s2.2333255338001416	s0.3896887351915142	s66.0124832447226
016_13	s1.3389248335448047	s0.4499845869493922	s246.01380904805524
016_14	s1.0911025620574524	s0.3906928135014459	s67.37682263830953
016_15	s2.232929472228281	s2.193612866762983	s67.36912070483794
"""

if __name__ == "__main__":
    print("Loading initial state...")
    start_state = parse_input(raw_input_data)
    
    # Run the "Hydraulic Press" logic
    # It tries to compress, then fix overlaps, then compress again.
    final_state = run_hydraulic_press(start_state, cycles=150)
    
    print("\nPressing Complete.")
    print(f"Final Area: {final_state.square_size**2}")
    plot_final(final_state)