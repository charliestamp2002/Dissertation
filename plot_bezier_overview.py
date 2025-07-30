import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting_refined import plot_all_bezier_clusters_overview

# ====================================================================
# CONFIGURATION
# ====================================================================

# Display settings
max_runs_per_cluster = 0  # Number of sample runs to show per cluster
show_control_points = True  # Show the 4 control points for each cluster
show_center_curves = True   # Show the center Bézier curve for each cluster
save_plot = True # Set to True to save the plot
save_path = "outputs/plots/bezier_clusters_overview.pdf"

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

# ====================================================================
# CREATE OVERVIEW PLOT
# ====================================================================

print(f"\nCreating Bézier clusters overview plot...")

fig = plot_all_bezier_clusters_overview(
    final_runs_df=final_runs_df,
    assignments_zones=assignments_df,
    cluster_control_points=cluster_control_points,
    max_runs_per_cluster=max_runs_per_cluster,
    show_control_points=show_control_points,
    show_center_curves=show_center_curves,
    title="Bézier Cluster Center Runs"
)

# ====================================================================
# SAVE OR SHOW
# ====================================================================

if save_plot:
    plt.figure(fig.number)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"\nPlot saved to: {save_path}")
else:
    plt.show()

print(f"\nOverview plot completed!")
print(f"Settings used:")
print(f"  - Max runs per cluster: {max_runs_per_cluster}")
print(f"  - Show control points: {show_control_points}")
print(f"  - Show center curves: {show_center_curves}")