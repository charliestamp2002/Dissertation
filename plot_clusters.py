import pandas as pd
import numpy as np
from plotting import plot_all_cluster_trajectories_on_pitch

# === CHOOSE CLUSTERING METHOD HERE ===
method = "bezier"  # Change to "autoencoder" to plot AE clusters
save_path = None   # Set to "outputs/bezier_clusters.png" to save instead of show
max_runs_per_cluster = 200


# === Load saved dataframes ===
print(f"Loading data for method: {method}")

final_runs_df = pd.read_parquet("outputs/final_runs_df.parquet")

if method == "bezier":
    assignments_df = pd.read_pickle("outputs/assignments_zones_bezier.pkl")
    cluster_control_points = np.load("outputs/bezier_cluster_control_points.npy", allow_pickle=True)
    bucket_pivot = pd.read_pickle("outputs/bezier_bucket_pivot.pkl")
    title = "Bézier+L1"
    is_autoencoder = False
else:
    assignments_df = pd.read_pickle("outputs/assignments_zones_ae.pkl")
    cluster_control_points = np.load("outputs/autoencoder_cluster_control_points.npy", allow_pickle=True)
    bucket_pivot = pd.read_pickle("outputs/ae_bucket_pivot.pkl")
    title = "Autoencoder"
    is_autoencoder = True

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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
else:
    import matplotlib.pyplot as plt
    plt.show()