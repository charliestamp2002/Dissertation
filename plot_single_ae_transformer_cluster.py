import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting_refined import plot_single_autoencoder_transformer_cluster

# ====================================================================
# CONFIGURATION
# ====================================================================

# Method and cluster to plot
method = "autoencoder"  # Options: "autoencoder", "transformer"
cluster_id = 15  # Change this to plot different clusters

# Center computation method
center_method = "mean"  # Options: "mean", "median", "medoid"
# - "mean": Average trajectory across all runs in cluster
# - "median": Element-wise median (more robust to outliers)
# - "medoid": Actual trajectory closest to the mean

# Display settings  
max_runs_to_show = 15  # Maximum number of runs to display
show_center_points = True  # Show points along the computed center trajectory
show_direction_arrows = True  # Show direction arrows on trajectories
save_plot = False  # Set to True to save the plot
save_path = f"outputs/{method}_cluster_{cluster_id}_detail.png"

# ====================================================================
# LOAD DATA
# ====================================================================

print(f"Loading {method} clustering data...")

# Load base data
final_runs_df = pd.read_parquet("outputs/final_runs_df.parquet")

# Load method-specific assignments
if method == "autoencoder":
    assignments_df = pd.read_pickle("outputs/assignments_zones_ae.pkl")
    method_name = "Autoencoder"
elif method == "transformer":
    assignments_df = pd.read_pickle("outputs/assignments_zones_transformer.pkl")
    method_name = "Transformer"
else:
    raise ValueError("Method must be 'autoencoder' or 'transformer'")

print(f"Loaded {len(final_runs_df)} run data points")
print(f"Loaded {len(assignments_df)} cluster assignments")

# Check if cluster exists
cluster_col = "ae_cluster" if method == "autoencoder" else "transformer_cluster"
if cluster_col not in assignments_df.columns and "assigned_cluster" in assignments_df.columns:
    cluster_col = "assigned_cluster"

cluster_runs = assignments_df[assignments_df[cluster_col] == cluster_id]
if len(cluster_runs) == 0:
    print(f"Error: No runs found for cluster {cluster_id}")
    print(f"Available clusters: {sorted(assignments_df[cluster_col].unique())}")
    exit()

# ====================================================================
# CREATE SINGLE CLUSTER PLOT
# ====================================================================

print(f"\nCreating detailed plot for {method_name} cluster {cluster_id}...")

fig = plot_single_autoencoder_transformer_cluster(
    final_runs_df=final_runs_df,
    assignments_zones=assignments_df,
    cluster_id=cluster_id,
    method_name=method_name,
    center_method=center_method,
    max_runs_to_show=max_runs_to_show,
    show_center_points=show_center_points,
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

print(f"\nSingle {method_name} cluster plot completed!")
print(f"Settings used:")
print(f"  - Method: {method_name}")
print(f"  - Cluster ID: {cluster_id}")
print(f"  - Center computation: {center_method}")
print(f"  - Max runs shown: {max_runs_to_show}")
print(f"  - Show center points: {show_center_points}")
print(f"  - Show direction arrows: {show_direction_arrows}")

# ====================================================================
# COMPARISON WITH OTHER CENTER METHODS
# ====================================================================

print(f"\n=== Comparison of Center Computation Methods ===")

other_methods = ["mean", "median", "medoid"]
for other_method in other_methods:
    if other_method != center_method:
        # Import the computation function to test other methods
        from plotting_refined import compute_cluster_center_trajectory
        
        other_center = compute_cluster_center_trajectory(
            final_runs_df, assignments_df, cluster_id, method=other_method
        )
        
        if other_center is not None:
            # Calculate trajectory length
            length = 0
            for i in range(len(other_center) - 1):
                dx = other_center[i+1, 0] - other_center[i, 0]
                dy = other_center[i+1, 1] - other_center[i, 1]
                length += np.sqrt(dx**2 + dy**2)
            
            print(f"  {other_method.capitalize()} center trajectory length: {length:.1f}m")
            print(f"    Start: ({other_center[0, 0]:.1f}, {other_center[0, 1]:.1f})")
            print(f"    End: ({other_center[-1, 0]:.1f}, {other_center[-1, 1]:.1f})")

print(f"\nNote: Different center computation methods can give different results:")
print(f"  - Mean: Smooth average, can be influenced by outliers")
print(f"  - Median: More robust to outliers, element-wise median")  
print(f"  - Medoid: Uses an actual trajectory from the cluster")