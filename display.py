import pandas as pd
import matplotlib.pyplot as plt
from shapely import affinity
from shapely.geometry import Polygon
import numpy as np

def get_tree_polygon(x, y, deg):
    """
    Creates a shapely Polygon for a Christmas tree at position (x, y) 
    with rotation 'deg' (in degrees).
    """
    # Dimensions defined in the Santa 2025 competition
    trunk_w = 0.15
    trunk_h = 0.2
    base_w = 0.7
    mid_w = 0.4
    top_w = 0.25
    tip_y = 0.8
    tier_1_y = 0.5
    tier_2_y = 0.25
    base_y = 0.0
    trunk_bottom_y = -trunk_h

    # Define the base polygon vertices (centered relative to its anchor)
    # The order is counter-clockwise starting from the top tip
    vertices = [
        (0, tip_y),                         # Tip
        (top_w / 2, tier_1_y),              # Right Top Tier Outer
        (top_w / 4, tier_1_y),              # Right Top Tier Inner
        (mid_w / 2, tier_2_y),              # Right Mid Tier Outer
        (mid_w / 4, tier_2_y),              # Right Mid Tier Inner
        (base_w / 2, base_y),               # Right Base Outer
        (trunk_w / 2, base_y),              # Right Trunk Top
        (trunk_w / 2, trunk_bottom_y),      # Right Trunk Bottom
        (-trunk_w / 2, trunk_bottom_y),     # Left Trunk Bottom
        (-trunk_w / 2, base_y),             # Left Trunk Top
        (-base_w / 2, base_y),              # Left Base Outer
        (-mid_w / 4, tier_2_y),             # Left Mid Tier Inner
        (-mid_w / 2, tier_2_y),             # Left Mid Tier Outer
        (-top_w / 4, tier_1_y),             # Left Top Tier Inner
        (-top_w / 2, tier_1_y)              # Left Top Tier Outer
    ]
    
    poly = Polygon(vertices)
    
    # 1. Rotate
    # Shapely rotates counter-clockwise by default
    poly = affinity.rotate(poly, deg, origin=(0, 0))
    
    # 2. Translate to position
    poly = affinity.translate(poly, xoff=x, yoff=y)
    
    return poly

def visualize_packing(csv_file, N):
    """
    Reads the submission CSV and plots the packing for N trees.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    # Filter for the specific N (id format is usually "001_0", "002_0", etc.)
    # We look for ids starting with the padded string of N (e.g., "005_")
    prefix = f"{int(N):03d}_"
    subset = df[df['id'].str.startswith(prefix)].copy()

    if subset.empty:
        print(f"No entries found for N={N} in the submission file.")
        return

    print(f"Plotting {len(subset)} trees for N={N}...")

    # Clean data: remove 's' prefix if present and convert to float
    for col in ['x', 'y', 'deg']:
        if subset[col].dtype == object:
            subset[col] = subset[col].astype(str).str.replace('s', '', regex=False)
        subset[col] = subset[col].astype(float)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    polygons = []
    
    # Generate and plot each tree
    for _, row in subset.iterrows():
        poly = get_tree_polygon(row['x'], row['y'], row['deg'])
        polygons.append(poly)
        
        # Plotting the polygon
        x_coords, y_coords = poly.exterior.xy
        ax.fill(x_coords, y_coords, alpha=0.6, fc='forestgreen', ec='darkgreen', linewidth=1)

    # Calculate bounding box of all trees
    all_x = []
    all_y = []
    for p in polygons:
        minx, miny, maxx, maxy = p.bounds
        all_x.extend([minx, maxx])
        all_y.extend([miny, maxy])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Determine the bounding square side length (score metric)
    width = max_x - min_x
    height = max_y - min_y
    bounding_side = max(width, height)
    
    # Draw the bounding box (approximated for visualization)
    # Note: The actual competition metric optimizes the square size, 
    # usually centered or anchored. Here we draw the tightest box around the plotted trees.
    rect = plt.Rectangle((min_x, min_y), width, height, 
                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # Set limits with some padding
    padding = 1.0
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    plt.title(f"Santa 2025: Packing for N={N}\nBounding Box Side ≈ {bounding_side:.4f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# --- Usage Example ---
if __name__ == "__main__":
    # Replace 'submission.csv' with your actual file path
    csv_path = input("Enter the path to your submission csv file (e.g., submission.csv): ").strip()
    n_input = input("Enter the number N to visualize (e.g., 5): ").strip()
    
    visualize_packing(csv_path, n_input)
