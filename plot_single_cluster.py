import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting_refined import plot_single_bezier_cluster

# ====================================================================
# CONFIGURATION
# ====================================================================

# Cluster to plot
cluster_id = 27  # Change this to plot different clusters

# Display settings  
max_runs_to_show = 45  # Maximum number of runs to display
show_control_points = True  # Show the 4 numbered control points
show_direction_arrows = True  # Show direction arrows on trajectories
save_plot = False  # Set to True to save the plot
save_path = f"outputs/bezier_cluster_{cluster_id}_detail.png"

# ====================================================================
# LOAD DATA
# ====================================================================

print("Loading Bézier clustering data...")

# Load required data files
final_runs_df = pd.read_parquet("outputs/final_runs_df.parquet")
assignments_df = pd.read_pickle("outputs/assignments_zones_bezier.pkl")
cluster_control_points = np.load("outputs/bezier_cluster_control_points.npy", allow_pickle=True)

print(f"Loaded {len(final_runs_df)} run data points")
print(f"Loaded {len(assignments_df)} cluster assignments")
print(f"Loaded {len(cluster_control_points)} cluster control points")

# Check if cluster exists
cluster_runs = assignments_df[assignments_df["assigned_cluster"] == cluster_id]
if len(cluster_runs) == 0:
    print(f"Error: No runs found for cluster {cluster_id}")
    print(f"Available clusters: {sorted(assignments_df['assigned_cluster'].unique())}")
    exit()

# ====================================================================
# CREATE SINGLE CLUSTER PLOT
# ====================================================================

print(f"\nCreating detailed plot for cluster {cluster_id}...")

fig = plot_single_bezier_cluster(
    final_runs_df=final_runs_df,
    assignments_zones=assignments_df,
    cluster_control_points=cluster_control_points,
    cluster_id=cluster_id,
    max_runs_to_show=max_runs_to_show,
    show_control_points=show_control_points,
    show_direction_arrows=show_direction_arrows
)

if fig is None:
    print("Failed to create plot")
    exit()

# ====================================================================
# SAVE OR SHOW
# ====================================================================

if save_plot:
    plt.figure(fig.number)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"\nPlot saved to: {save_path}")
else:
    plt.show()

print(f"\nSingle cluster plot completed!")
print(f"Settings used:")
print(f"  - Cluster ID: {cluster_id}")
print(f"  - Max runs shown: {max_runs_to_show}")
print(f"  - Show control points: {show_control_points}")
print(f"  - Show direction arrows: {show_direction_arrows}")

# ====================================================================
# ADDITIONAL CLUSTER INFORMATION
# ====================================================================

print(f"\nCluster {cluster_id} Summary:")
total_runs = len(cluster_runs)
print(f"  Total runs in this cluster: {total_runs}")

if cluster_control_points[cluster_id] is not None:
    control_pts = cluster_control_points[cluster_id]
    print(f"  Number of control points: {len(control_pts)}")
    
    # Calculate some basic trajectory statistics
    center_curve_length = 0
    for i in range(len(control_pts) - 1):
        dx = control_pts[i+1, 0] - control_pts[i, 0]
        dy = control_pts[i+1, 1] - control_pts[i, 1]
        center_curve_length += np.sqrt(dx**2 + dy**2)
    
    print(f"  Approximate center curve length: {center_curve_length:.1f}m")
    print(f"  Start position: ({control_pts[0, 0]:.1f}, {control_pts[0, 1]:.1f})")
    print(f"  End position: ({control_pts[-1, 0]:.1f}, {control_pts[-1, 1]:.1f})")
    
    # Calculate general direction
    overall_dx = control_pts[-1, 0] - control_pts[0, 0]
    overall_dy = control_pts[-1, 1] - control_pts[0, 1]
    
    if abs(overall_dx) > abs(overall_dy):
        direction = "forward" if overall_dx > 0 else "backward"
    else:
        direction = "toward top of pitch" if overall_dy > 0 else "toward bottom of pitch"
    
    print(f"  General direction: {direction}")