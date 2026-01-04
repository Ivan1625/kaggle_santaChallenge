import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import warnings
import os
import glob
import argparse
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Decimal precision and scaling factor
getcontext().prec = 50
scale_factor = Decimal('1e30')

# Default file paths
INPUT_FILES = [
    r"C:/Users/user/Downloads/submission - 2026-01-05T015426.248.csv",
    r"C:/Users/user/Downloads/submission - 2026-01-05T012456.891.csv",
    r"C:/Users/user/Downloads/submission - 2026-01-05T012449.517.csv",
    r"C:/Users/user/Downloads/submission - 2026-01-05T012440.706.csv",
    r"C:/Users/user/Downloads/best_submission (2).csv",
    r"C:/Users/user/Downloads/submission - 2026-01-05T015435.472.csv",
    r"C:/Users/user/Downloads/submission - 2026-01-05T015441.147.csv"
]
OUTPUT_FILE = r"C:/Users/user/Downloads/merged_submission.csv"

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

    except Exception as e:
        print(f"Error calculating score: {e}")
        return float('inf')

def parse_args():
    parser = argparse.ArgumentParser(description='Merge multiple Christmas tree packing submissions')
    parser.add_argument('-i', '--input', nargs='+', help='Input CSV files to merge')
    parser.add_argument('-p', '--pattern', help='Pattern to search for input files (e.g., "*.csv")')
    parser.add_argument('-d', '--directory', help='Directory to search for input files')
    parser.add_argument('-o', '--output', help='Output CSV file')
    return parser.parse_args()

def merge_multiple_submissions(input_files, output_path):
    """Merge multiple submission files, taking the best configuration for each group."""
    if not input_files:
        print("No input files specified.")
        return
    
    print(f"Processing {len(input_files)} input files:")
    for file in input_files:
        print(f" - {file}")
    
    # Load all submissions
    dataframes = []
    file_names = []
    
    for file_path in input_files:
        try:
            print(f"Reading {file_path}...")
            df = pd.read_csv(file_path)
            # Extract group ID
            df['group_id'] = df['id'].astype(str).str.split('_').str[0]
            dataframes.append(df)
            file_names.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not dataframes:
        print("No valid submissions loaded.")
        return
    
    # Get all unique group IDs
    all_groups = set()
    for df in dataframes:
        all_groups.update(df['group_id'].unique())
    
    unique_groups = sorted(list(all_groups), key=lambda x: int(x))
    
    best_rows = []
    source_counts = {name: 0 for name in file_names}
    total_scores = {name: 0.0 for name in file_names}
    valid_flags = {name: True for name in file_names}
    total_score_merged = 0.0
    
    print(f"\nProcessing {len(unique_groups)} groups...")
    
    for gid in tqdm(unique_groups):
        best_score = float('inf')
        best_df = None
        best_source = None
        
        for i, df in enumerate(dataframes):
            if gid in df['group_id'].values:
                sub_df = df[df['group_id'] == gid].copy()
                score = calculate_group_score(sub_df)
                
                # Update source stats
                source_name = file_names[i]
                if score == float('inf'):
                    valid_flags[source_name] = False
                else:
                    total_scores[source_name] += score
                
                # Keep track of the best configuration
                if score < best_score:
                    best_score = score
                    best_df = sub_df
                    best_source = source_name
        
        if best_df is not None:
            best_rows.append(best_df)
            if best_source:
                source_counts[best_source] += 1
                if best_score != float('inf'):
                    total_score_merged += best_score
    
    # Compile final dataframe
    if best_rows:
        result_df = pd.concat(best_rows, ignore_index=True)
        result_df = result_df.drop(columns=['group_id'])
        
        # Save
        result_df.to_csv(output_path, index=False)
        
        # Reporting
        print("\n" + "="*60)
        print("                     MERGE RESULTS                      ")
        print("="*60)
        
        print("Groups taken from each file:")
        for name, count in source_counts.items():
            print(f" - {name}: {count} groups")
        print("-" * 60)
        
        print("Scores by file:")
        for name, score in total_scores.items():
            status = f"{score:.6f}" if valid_flags[name] else "INVALID (Overlaps)"
            print(f" - {name}: {status}")
        print("-" * 60)
        
        print(f"Score of merged result: {total_score_merged:.6f}")
        
        # Calculate Improvement
        valid_scores = [score for name, score in total_scores.items() if valid_flags[name]]
        if valid_scores:
            best_input_score = min(valid_scores)
            improvement = best_input_score - total_score_merged
            
            if improvement > 0:
                print(f"SUCCESS: Score improved by {improvement:.6f}")
            elif improvement == 0:
                print("RESULT: No improvement (Merged file equals best input).")
            else:
                print(f"WARNING: Score degraded by {-improvement:.6f}")
        else:
            print("RESULT: All input files had invalid configurations. Merged file fixed collisions.")
        
        print("="*60)
        print(f"Output saved to: {output_path}")
    else:
        print("No valid configurations found in any input file.")

def main():
    args = parse_args()
    
    # Determine input files
    input_files = INPUT_FILES  # Default
    if args.input:
        input_files = args.input
    elif args.pattern:
        directory = args.directory or '.'
        input_files = glob.glob(os.path.join(directory, args.pattern))
    
    # Determine output file
    output_file = args.output or OUTPUT_FILE
    
    merge_multiple_submissions(input_files, output_file)

if __name__ == "__main__":
    main()
