import pandas as pd
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from tqdm import tqdm
import math

# ==========================================
# 1. GEOMETRY DEFINITION (From Inversion's Notebook)
# ==========================================
def get_tree_polygon():
    # Dimensions from the "Getting Started" notebook code
    trunk_w = 0.15
    trunk_h = 0.2
    base_w = 0.7
    mid_w = 0.4
    top_w = 0.25
    tip_y = 0.8
    tier_1_y = 0.5
    tier_2_y = 0.25
    base_y = 0.0
    
    # Vertices defined counter-clockwise
    coords = [
        (0.0, tip_y),                         # Tip
        (top_w / 2, tier_1_y),                # Top Tier Right
        (top_w / 4, tier_1_y),                # Indent
        (mid_w / 2, tier_2_y),                # Mid Tier Right
        (mid_w / 4, tier_2_y),                # Indent
        (base_w / 2, base_y),                 # Base Right
        (trunk_w / 2, base_y),                # Trunk Top Right
        (trunk_w / 2, -trunk_h),              # Trunk Bottom Right
        (-trunk_w / 2, -trunk_h),             # Trunk Bottom Left
        (-trunk_w / 2, base_y),               # Trunk Top Left
        (-base_w / 2, base_y),                # Base Left
        (-mid_w / 4, tier_2_y),               # Indent
        (-mid_w / 2, tier_2_y),               # Mid Tier Left
        (-top_w / 4, tier_1_y),               # Indent
        (-top_w / 2, tier_1_y),               # Top Tier Left
        (0.0, tip_y)                          # Close loop
    ]
    return Polygon(coords)

# Global base tree to avoid recreating it
BASE_TREE = get_tree_polygon()

def create_poly(x, y, deg):
    """
    Transforms the base tree polygon to the specified location and rotation.
    Note: Shapely rotates counter-clockwise by default.
    """
    # Rotate around (0,0) - the definition point of the tree
    # The competition uses degrees.
    poly = affinity.rotate(BASE_TREE, deg, origin=(0, 0))
    poly = affinity.translate(poly, x, y)
    return poly

# ==========================================
# 2. PROCESSING LOGIC
# ==========================================

def process_submission(input_file, output_file):
    print(f"Reading {input_file}...")
    
    # Read as string to handle 's' prefixes manually
    try:
        df = pd.read_csv(input_file, dtype=str)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return

    # 1. CLEAN DATA (Remove 's' for calculations)
    # We strip 's' if it exists, otherwise keep the number
    df['x_num'] = df['x'].astype(str).str.replace('s', '', regex=False).astype(float)
    df['y_num'] = df['y'].astype(str).str.replace('s', '', regex=False).astype(float)
    df['deg_num'] = df['deg'].astype(str).str.replace('s', '', regex=False).astype(float)

    # Parse Puzzle ID (format: N_i)
    try:
        df['puzzle_n'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    except:
        print("Error: IDs must be in format 'N_index' (e.g., 1_0, 5_2).")
        return

    unique_puzzles = sorted(df['puzzle_n'].unique())
    
    collisions_count = 0
    total_score = 0.0
    
    print(f"Validating {len(unique_puzzles)} puzzles and calculating score...")
    
    for n in tqdm(unique_puzzles):
        subset = df[df['puzzle_n'] == n]
        
        # Verify we have exactly n trees for puzzle n (sanity check)
        if len(subset) != n:
            print(f"Warning: Puzzle {n} should have {n} trees, found {len(subset)}.")

        polys = []
        # Create polygons
        for _, row in subset.iterrows():
            p = create_poly(row['x_num'], row['y_num'], row['deg_num'])
            polys.append(p)
            
        # --- COLLISION CHECK (Inversion's Algorithm) ---
        # Using STRtree for efficient spatial indexing
        if len(polys) > 1:
            tree = STRtree(polys)
            # query returns indices of all geometries that intersect the envelope of p
            for i, p in enumerate(polys):
                # This query is approximate (bounding box check)
                candidates = tree.query(p)
                for j in candidates:
                    # Don't check against self or pairs we've likely checked (j > i)
                    # But standard practice is usually just i != j. 
                    # To allow touching, we check intersection AND NOT touching.
                    if i < j: 
                        p2 = polys[j]
                        # Exact check: Overlap with area > 0 (intersects) but NOT just touching edges
                        if p.intersects(p2) and not p.touches(p2):
                            # Double check area to avoid float precision false positives on touching
                            if p.intersection(p2).area > 1e-9:
                                print(f"!! COLLISION in Puzzle {n}: Tree {i} overlaps Tree {j}")
                                collisions_count += 1

        # --- SCORE CALCULATION ---
        # Score = (Max Side Length)^2 / N
        if not polys:
            continue
            
        minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
        for p in polys:
            b = p.bounds # (minx, miny, maxx, maxy)
            minx = min(minx, b[0])
            miny = min(miny, b[1])
            maxx = max(maxx, b[2])
            maxy = max(maxy, b[3])
            
        width = maxx - minx
        height = maxy - miny
        side = max(width, height)
        
        puzzle_score = (side ** 2) / n
        total_score += puzzle_score

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("                 REPORT")
    print("="*50)
    
    if collisions_count == 0:
        print("✅ VALIDITY: Passed (0 Collisions)")
    else:
        print(f"❌ VALIDITY: FAILED ({collisions_count} Collisions detected)")
        
    print(f"🏆 TOTAL SCORE: {total_score:.20f}")
    print("="*50)

    # 2. SAVE FIXED FILE
    # Ensures every coordinate starts with 's' to satisfy Kaggle parser
    print(f"Saving fixed format to {output_file}...")
    
    def add_s(val):
        s = str(val)
        return s if s.startswith('s') else f"s{s}"

    # Apply formatting
    df['x'] = df['x_num'].apply(add_s)
    df['y'] = df['y_num'].apply(add_s)
    df['deg'] = df['deg_num'].apply(add_s)
    
    # Save submission
    final_df = df[['id', 'x', 'y', 'deg']]
    final_df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    INPUT_CSV = r'C:\Users\user\Downloads\submission (126).csv'       # Your messy file
    OUTPUT_CSV = r'C:\Users\user\Downloads\submission (127).csv' # The file to upload
    process_submission(INPUT_CSV, OUTPUT_CSV)