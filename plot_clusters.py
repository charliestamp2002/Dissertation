import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from plotting import plot_all_cluster_trajectories_on_pitch, draw_pitch
from bezier_clustering import resample_coords, fit_bezier_curve, evaluate_bezier_curve

# === CHOOSE CLUSTERING METHOD HERE ===
method = "bezier"  # Change to "autoencoder" to plot AE clusters
save_path = None   # Set to "outputs/bezier_clusters.png" to save instead of show
max_runs_per_cluster = 40

# === SINGLE PITCH OVERVIEW PLOT ===
plot_all_on_single_pitch = True # Set to True to plot all clusters on one pitch
single_pitch_max_runs_per_cluster = 5 # Limit runs per cluster for readability
show_cluster_centers = True  # Show cluster control points/centers


def plot_all_clusters_single_pitch(
    final_runs_df,
    assignments_zones,
    cluster_control_points,
    max_runs_per_cluster=5,
    is_autoencoder=False,
    show_centers=True,
    title="All Clusters Overview"
):
    """
    Plot all clusters on a single pitch with different colors
    """
    
    # Set up colors for clusters
    import matplotlib.cm as cm
    num_clusters = len(cluster_control_points)
    colors = cm.tab20(np.linspace(0, 1, min(20, num_clusters)))  # Use tab20 colormap
    if num_clusters > 20:
        # If more than 20 clusters, cycle through colors
        colors = [colors[i % 20] for i in range(num_clusters)]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_pitch(ax)
    
    print(f"Plotting {num_clusters} clusters on single pitch...")
    
    for cluster_idx in range(num_clusters):
        color = colors[cluster_idx]
        
        # Get runs for this cluster
        cluster_run_ids = assignments_zones.loc[
            assignments_zones["assigned_cluster"] == cluster_idx, "run_id"].tolist()
        
        if len(cluster_run_ids) == 0:
            continue
            
        # Limit number of runs for readability
        if len(cluster_run_ids) > max_runs_per_cluster:
            cluster_run_ids = random.sample(cluster_run_ids, max_runs_per_cluster)
        
        # Plot trajectories for this cluster
        for run_id in cluster_run_ids:
            run_df = final_runs_df[final_runs_df["run_id"] == run_id]
            if len(run_df) == 0:
                continue
                
            coords = run_df[["x_mirror_c", "y_mirror_c"]].values
            if coords.shape[0] < 2:
                continue

            # Create trajectory
            if is_autoencoder:
                trajectory = coords
            else:
                resampled = resample_coords(coords, num_points=50)
                if len(resampled) >= 4:
                    control_pts = fit_bezier_curve(resampled, num_control_points=4)
                    trajectory = evaluate_bezier_curve(control_pts, num_points=50)
                else:
                    trajectory = coords

            # Plot trajectory with cluster-specific color
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   alpha=0.4, color=color, linewidth=0.8)
        
        # Plot cluster center/control points if available and requested
        if show_centers and not is_autoencoder:
            if cluster_control_points[cluster_idx] is not None:
                control_pts = cluster_control_points[cluster_idx]
                if len(control_pts) >= 2:
                    # Plot control points
                    ax.scatter(control_pts[:, 0], control_pts[:, 1], 
                             c=[color], s=30, marker='X', alpha=0.8, 
                             edgecolors='black', linewidths=0.5)
                    
                    # Connect control points to show cluster center curve
                    center_curve = evaluate_bezier_curve(control_pts, num_points=50)
                    ax.plot(center_curve[:, 0], center_curve[:, 1], 
                           color='black', linewidth=2, alpha=0.7)
    
    ax.set_title(f"{title} - All {num_clusters} Clusters", fontsize=14, fontweight='bold')
    
    # Add legend showing number of clusters
    ax.text(0.02, 0.98, f"Clusters: {num_clusters}\nRuns per cluster: ≤{max_runs_per_cluster}", 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig


# === Load saved dataframes ===
print(f"Loading data for method: {method}")

final_runs_df = pd.read_parquet("outputs/final_runs_df.parquet")

if method == "bezier":
    assignments_df = pd.read_pickle("outputs/assignments_zones_bezier.pkl")
    cluster_control_points = np.load("outputs/bezier_cluster_control_points.npy", allow_pickle=True)
    bucket_pivot = pd.read_pickle("outputs/bezier_bucket_pivot.pkl")
    title = "Bézier+L1"
    is_autoencoder = False

elif method == "autoencoder":
    assignments_df = pd.read_pickle("outputs/assignments_zones_ae.pkl")
    cluster_control_points = np.load("outputs/autoencoder_cluster_control_points.npy", allow_pickle=True)
    bucket_pivot = pd.read_pickle("outputs/ae_bucket_pivot.pkl")
    title = "Autoencoder"
    is_autoencoder = True

elif method == "transformer":
    assignments_df = pd.read_pickle("outputs/assignments_zones_transformer.pkl")
    cluster_control_points = np.load("outputs/transformer_cluster_control_points.npy", allow_pickle=True)
    bucket_pivot = pd.read_pickle("outputs/transformer_bucket_pivot.pkl")
    title = "Transformer"
    is_autoencoder = True  # Transformer doesn't use control points, treat like AE
else:
    raise ValueError(f"Unsupported method: {method}")

# === Plot all clusters ===
plot_all_cluster_trajectories_on_pitch(
    final_runs_df=final_runs_df,
    assignments_zones=assignments_df,
    cluster_control_points=cluster_control_points,
    bucket_pivot=bucket_pivot,
    num_control_points=4,
    max_runs_per_cluster=max_runs_per_cluster,
    title=title,
    is_autoencoder=is_autoencoder,
    plot_absolute_positions=False,
)

# === Save or Show ===
if save_path:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
else:
    import matplotlib.pyplot as plt
    plt.show()

# === SINGLE PITCH OVERVIEW PLOT ===
if plot_all_on_single_pitch:
    print(f"\n=== CREATING SINGLE PITCH OVERVIEW ===")
    
    single_pitch_fig = plot_all_clusters_single_pitch(
        final_runs_df=final_runs_df,
        assignments_zones=assignments_df,
        cluster_control_points=cluster_control_points,
        max_runs_per_cluster=single_pitch_max_runs_per_cluster,
        is_autoencoder=is_autoencoder,
        show_centers=show_cluster_centers,
        title=title
    )
    
    # Save or show single pitch plot
    if save_path:
        single_pitch_save_path = save_path.replace(".png", "_single_pitch.png")
        plt.figure(single_pitch_fig.number)
        plt.savefig(single_pitch_save_path, dpi=300, bbox_inches="tight")
        print(f"Single pitch overview saved to: {single_pitch_save_path}")
    else:
        plt.show()
        
    print(f"Single pitch overview completed!")
    print(f"  - Plotted {len(cluster_control_points)} clusters")
    print(f"  - Max {single_pitch_max_runs_per_cluster} runs per cluster")
    print(f"  - Cluster centers shown: {show_cluster_centers}")