import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting_refined import draw_pitch, get_cluster_colors, compute_cluster_center_trajectory
import random

# ====================================================================
# CONFIGURATION
# ====================================================================

# Method to plot
method = "transformer"  # Options: "autoencoder", "transformer"

# Display settings for all subplots
max_runs_per_cluster = 0  # Number of sample runs to show per cluster (0 = none)
show_center_trajectories = True  # Show the computed center trajectories
show_center_points = True  # Show points along center trajectories for comparison
save_plot = True  # Set to True to save the plot
save_path = f"outputs/plots/{method}_clusters_overview_comparison.pdf"

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

# Check cluster column
cluster_col = "ae_cluster" if method == "autoencoder" else "transformer_cluster"
if cluster_col not in assignments_df.columns and "assigned_cluster" in assignments_df.columns:
    cluster_col = "assigned_cluster"

print(f"Found {assignments_df[cluster_col].nunique()} unique clusters")

# ====================================================================
# CREATE COMPARISON SUBPLOT
# ====================================================================

print(f"\nCreating comparison plot for all {method_name} cluster centers...")
print("Showing Mean, Median, and Medoid methods side by side...")

# Create subplot figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
center_methods = ["mean", "median", "medoid"]

# Get cluster column name
cluster_col = "ae_cluster" if method == "autoencoder" else "transformer_cluster"
if cluster_col not in assignments_df.columns and "assigned_cluster" in assignments_df.columns:
    cluster_col = "assigned_cluster"

# Get all unique clusters
unique_clusters = sorted(assignments_df[cluster_col].unique())
num_clusters = len(unique_clusters)

# Get professional colors for clusters (consistent across subplots)
colors = get_cluster_colors(num_clusters)

for subplot_idx, center_method in enumerate(center_methods):
    ax = axes[subplot_idx]
    draw_pitch(ax)
    
    print(f"  Computing {center_method} centers...")
    
    successful_centers = 0
    
    for i, cluster_id in enumerate(unique_clusters):
        color = colors[i % len(colors)]
        
        # Get runs for this cluster
        cluster_runs = assignments_df[assignments_df[cluster_col] == cluster_id]
        cluster_run_ids = cluster_runs["run_id"].tolist()
        
        if len(cluster_run_ids) == 0:
            continue
        
        # Sample runs for display if requested
        if max_runs_per_cluster > 0:
            if len(cluster_run_ids) > max_runs_per_cluster:
                cluster_run_ids = random.sample(cluster_run_ids, max_runs_per_cluster)
            
            # Plot sample trajectories for this cluster
            for run_id in cluster_run_ids:
                run_df = final_runs_df[final_runs_df["run_id"] == run_id]
                if len(run_df) == 0:
                    continue
                    
                coords = run_df[["x_mirror_c", "y_mirror_c"]].values
                if coords.shape[0] < 2:
                    continue

                # Plot trajectory with very low alpha for background context
                ax.plot(coords[:, 0], coords[:, 1], 
                       alpha=0.1, color=color, linewidth=0.5, zorder=1)
        
        # Compute and plot cluster center trajectory
        center_trajectory = compute_cluster_center_trajectory(
            final_runs_df, assignments_df, cluster_id, method=center_method
        )
        
        if center_trajectory is not None:
            successful_centers += 1
            
            if show_center_trajectories:
                # Plot center trajectory
                ax.plot(center_trajectory[:, 0], center_trajectory[:, 1], 
                       color=color, linewidth=2.0, alpha=0.8, zorder=3)
            
            if show_center_points:
                # Plot center points (every nth point)
                step = max(1, len(center_trajectory) // 8)  # Show ~8 points per trajectory
                center_points = center_trajectory[::step]
                ax.scatter(center_points[:, 0], center_points[:, 1], 
                         c=[color], s=15, marker='o', alpha=0.7, 
                         edgecolors='white', linewidths=0.5, zorder=4)
    
    # Set subplot title
    ax.set_title(f"{center_method.capitalize()} Method", 
                fontsize=12, fontweight='bold', pad=10)
    
    print(f"{successful_centers} centers computed successfully")

# Add main title
fig.suptitle(f"{method_name} Cluster Centers Comparison - All Methods", 
            fontsize=16, fontweight='bold', y=0.95)

# Add legend (only on the rightmost subplot)
legend_elements = []
if show_center_trajectories:
    legend_elements.append(
        plt.Line2D([0], [0], color='#2E86AB', linewidth=2.0, alpha=0.8,
                  label='Cluster center trajectories')
    )
if show_center_points:
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#2E86AB', markersize=6,
                  markeredgecolor='white', markeredgewidth=0.5,
                  label='Center trajectory points', linestyle='None')
    )
if max_runs_per_cluster > 0:
    legend_elements.append(
        plt.Line2D([0], [0], color='#2E86AB', alpha=0.1, linewidth=0.5,
                  label='Sample trajectories')
    )

if legend_elements:
    axes[2].legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98), fontsize=9,
                  fancybox=True, shadow=True, framealpha=0.9)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

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

print(f"\n{method_name} clusters comparison plot completed!")
print(f"Settings used:")
print(f"  - Method: {method_name}")
print(f"  - Center computation methods: Mean, Median, Medoid (all shown)")
print(f"  - Show center trajectories: {show_center_trajectories}")
print(f"  - Show center points: {show_center_points}")
print(f"  - Sample runs per cluster: {max_runs_per_cluster}")

# ====================================================================
# ADDITIONAL ANALYSIS
# ====================================================================

print(f"\n=== {method_name} Cluster Analysis ===")

# Get some basic statistics about the clusters
cluster_sizes = assignments_df[cluster_col].value_counts().sort_index()

print(f"\nCluster Size Distribution:")
print(f"  Smallest cluster: {cluster_sizes.min()} runs")
print(f"  Largest cluster: {cluster_sizes.max()} runs")
print(f"  Average cluster size: {cluster_sizes.mean():.1f} runs")
print(f"  Median cluster size: {cluster_sizes.median():.1f} runs")

# Show top 5 largest clusters
print(f"\nTop 5 Largest Clusters:")
top_clusters = cluster_sizes.nlargest(5)
for cluster_id, size in top_clusters.items():
    print(f"  Cluster {cluster_id}: {size} runs")

# Show clusters with few runs
small_clusters = cluster_sizes[cluster_sizes <= 5]
if len(small_clusters) > 0:
    print(f"\nClusters with ≤5 runs: {len(small_clusters)} clusters")
    print(f"  IDs: {sorted(small_clusters.index.tolist())}")

print(f"\nNote: This overview shows the computed center trajectory for each cluster.")
print(f"The center trajectories represent the 'average' movement pattern for each cluster,")
print(f"computed using the {center_method} method across all runs in that cluster.")