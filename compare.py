import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. METRIC & GEOMETRY CLASSES (FROM KAGGLE)
# ==========================================

# Decimal precision and scaling factor
getcontext().prec = 25
scale_factor = Decimal('1e18')

class ParticipantVisibleError(Exception):
    pass

class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))

def calculate_group_score(df_group):
    """
    Calculates the score for a specific group of trees (N packing).
    Returns float('inf') if there is a collision or error.
    """
    try:
        # 1. Clean data (Remove 's' prefix for calculation)
        local_df = df_group.copy()
        cols = ['x', 'y', 'deg']
        for c in cols:
            local_df[c] = local_df[c].astype(str).str.replace('s', '', regex=False)
        
        num_trees = len(local_df)
        
        # 2. Create Tree Objects
        placed_trees = []
        for _, row in local_df.iterrows():
            placed_trees.append(ChristmasTree(row['x'], row['y'], row['deg']))

        # 3. Collision Detection
        all_polygons = [p.polygon for p in placed_trees]
        r_tree = STRtree(all_polygons)

        for i, poly in enumerate(all_polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i: continue
                # Intersects but does not just touch
                if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                    return float('inf')

        # 4. Calculate Bounds and Score
        bounds = unary_union(all_polygons).bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        side_length_scaled = max(width, height)

        score = (Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(num_trees)
        return float(score)

    except Exception:
        return float('inf')

# ==========================================
# 2. MERGING LOGIC & STATS
# ==========================================

def merge_submission_files(file_path_a, file_path_b, output_path):
    print(f"Reading {file_path_a}...")
    df_a = pd.read_csv(file_path_a)
    print(f"Reading {file_path_b}...")
    df_b = pd.read_csv(file_path_b)

    # Extract group ID
    df_a['group_id'] = df_a['id'].astype(str).str.split('_').str[0]
    df_b['group_id'] = df_b['id'].astype(str).str.split('_').str[0]

    unique_groups = df_a['group_id'].unique()
    
    best_rows = []
    
    # Stats Counters
    count_a = 0
    count_b = 0
    total_score_a = 0.0
    total_score_b = 0.0
    total_score_merged = 0.0
    
    a_valid = True
    b_valid = True
    
    print(f"\nProcessing {len(unique_groups)} groups...")
    
    for gid in unique_groups:
        sub_a = df_a[df_a['group_id'] == gid].copy()
        sub_b = df_b[df_b['group_id'] == gid].copy()
        
        # Calculate scores
        score_a = calculate_group_score(sub_a)
        score_b = calculate_group_score(sub_b)
        
        # Accumulate totals (Check for validity)
        if score_a == float('inf'):
            a_valid = False
        else:
            total_score_a += score_a
            
        if score_b == float('inf'):
            b_valid = False
        else:
            total_score_b += score_b
        
        # Comparison Logic
        if score_a <= score_b:
            best_rows.append(sub_a)
            count_a += 1
            total_score_merged += score_a
        else:
            best_rows.append(sub_b)
            count_b += 1
            total_score_merged += score_b

    # Compile final dataframe
    result_df = pd.concat(best_rows, ignore_index=True)
    result_df = result_df.drop(columns=['group_id'])
    
    # Save
    result_df.to_csv(output_path, index=False)
    
    # ==========================================
    # 3. REPORTING
    # ==========================================
    print("\n" + "="*40)
    print("           MERGE RESULTS            ")
    print("="*40)
    
    print(f"Groups taken from File A : {count_a}")
    print(f"Groups taken from File B : {count_b}")
    print("-" * 40)
    
    # Report Status
    txt_a = f"{total_score_a:.6f}" if a_valid else "INVALID (Overlaps)"
    txt_b = f"{total_score_b:.6f}" if b_valid else "INVALID (Overlaps)"
    txt_m = f"{total_score_merged:.6f}"
    
    print(f"Score File A             : {txt_a}")
    print(f"Score File B             : {txt_b}")
    print(f"Score Best Merged        : {txt_m}")
    print("-" * 40)
    
    # Calculate Improvement
    # We compare against the best VALID input file.
    inputs = []
    if a_valid: inputs.append(total_score_a)
    if b_valid: inputs.append(total_score_b)
    
    if not inputs:
        print("RESULT: Both input files were invalid. Merged file fixed collisions.")
    else:
        best_input_score = min(inputs)
        improvement = best_input_score - total_score_merged
        
        if improvement > 0:
            print(f"SUCCESS: Score Improved by {improvement:.6f}")
        elif improvement == 0:
            print("RESULT: No improvement (Merged file equals best input).")
        else:
            # This handles rare edge cases where a collision might have been introduced 
            # (unlikely with this logic) or floating point drift.
            print(f"RESULT: Score change {improvement:.6f}")

    print("="*40)
    print(f"Output saved to: {output_path}")

# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == "__main__":
    # SPECIFIC PATHS
    FILE_A = r"C:/Users/user/Downloads/submission (fk rounding).csv"
    FILE_B = r"C:/Users/user/Downloads/submission_8_rotated.csv"
    OUTPUT_FILE = r"C:/Users/user/Downloads/submission_frankenstein.csv"
    
    if os.path.exists(FILE_A) and os.path.exists(FILE_B):
        merge_submission_files(FILE_A, FILE_B, OUTPUT_FILE)
    else:
        print("Error: Input files not found. Please check paths:")
        print(f"A: {FILE_A}")
        print(f"B: {FILE_B}")