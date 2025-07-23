import pandas as pd
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

from data_utils import (
    load_tracking_data,
    load_event_metadata,
    load_tracking_metadata,
    match_events_to_tracking,
    load_and_align_events,
    build_opta_metadata,
    merge_assignments_with_metadata,
    add_position_buckets,
    compute_position_pivot,
    compute_bucket_pivot,
    get_valid_resampled_tensor
)

from run_segmentation import (
    segment_runs,
    filter_off_ball_runs_with_distance,
    process_runs,
    extract_zone_and_run_features
)

from bezier_clustering import bezier_kmeans_clustering
from autoencoder_clustering import run_autoencoder_clustering
from transformer_clustering import run_transformer_clustering
from metrics import run_all_metrics


start_time = time.time()
# Optional: limit number of files for quick testing
# full number of files = 1127
N_FILES = 50

# Load tracking data
print("Loading tracking data...")
frames_df, players_df, used_match_ids = load_tracking_data(n_files=N_FILES, block_size=500, num_blocks=10, seed=42)
print(f"Loaded {len(frames_df)} frames and {len(players_df)} player entries from {len(used_match_ids)} matches.\n")

# Load event metadata
print("Loading StatsBomb event metadata...")
event_meta_df = load_event_metadata()
print(f"Loaded metadata from {len(event_meta_df)} event files.\n")

# Load tracking metadata
print("Loading Second Spectrum tracking metadata...")
tracking_meta_df = load_tracking_metadata()
print(f"Loaded tracking metadata for {len(tracking_meta_df)} matches.\n")

# Team code mapping
team_code_map = {
    'FUL': 'Fulham', 'BRE': 'Brentford', 'CRY': 'Crystal Palace',
    'TOT': 'Tottenham Hotspur', 'BOU': 'AFC Bournemouth', 'SOU': 'Southampton',
    'AVL': 'Aston Villa', 'WHU': 'West Ham United', 'MUN': 'Manchester United',
    'ARS': 'Arsenal', 'LEI': 'Leicester City', 'NEW': 'Newcastle United',
    'BHA': 'Brighton & Hove Albion', 'IPS': 'Ipswich Town', 'EVE': 'Everton',
    'LIV': 'Liverpool', 'LEE': 'Leeds United', 'NOT': 'Nottingham Forest',
    'MCI': 'Manchester City', 'WOL': 'Wolverhampton Wanderers',
    'SHU': 'Sheffield United', 'CHE': 'Chelsea', 'LUT': 'Luton Town',
    'BUR': 'Burnley'
}
print("Loaded team name mapping.\n")

# Match events to tracking data
print("Matching events to tracking metadata...")
event_tracking_df_clean = match_events_to_tracking(event_meta_df, tracking_meta_df, team_code_map)
print(f"Matched {len(event_tracking_df_clean)} event files to tracking metadata.\n")

# Show a few example rows
# print("Preview of matched event-tracking metadata:")
# print(event_tracking_df_clean.head())

# Align event frames
events_dfs_by_match = load_and_align_events(frames_df, players_df, event_tracking_df_clean)

# Preview one match
debug_match_id = "tracking_" + event_tracking_df_clean.iloc[0]["tracking_suffix"]
# print(f"\nPreviewing events for {debug_match_id}")
# print(events_dfs_by_match[debug_match_id].head())

# Attach player metadata
players_df, metadata_df, metadata_dict = build_opta_metadata(players_df, used_match_ids)

# print(players_df[["playerId", "player_name", "position", "team_role"]].dropna().head())
# print(metadata_dict)  # Surname → optaId

# 1. Segment runs from player data
runs_list = segment_runs(players_df)

# 2. Filter to off-ball runs
off_ball_runs = filter_off_ball_runs_with_distance(runs_list, frames_df, min_distance=3.0)

# 3. Annotate and normalize runs
adjusted_runs_list ,final_runs_df = process_runs(players_df, frames_df, off_ball_runs)

run_lengths = final_runs_df.groupby("run_id").size()


# print(final_runs_df.head())

# plt.hist(run_lengths, bins=30, edgecolor='black')
# plt.title("Distribution of Run Lengths (Number of Frames per Run)")
# plt.xlabel("Number of Frames")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()

zones_df = extract_zone_and_run_features(
    final_runs_df,
    players_df,
    frames_df,
    events_dfs_by_match,
    metadata_dict
)

print(zones_df.head())

bezier_cluster_control_points, bezier_assignments_df = bezier_kmeans_clustering(
    final_runs_df=final_runs_df,
    adjusted_runs_list=adjusted_runs_list,
    k_clusters=70,
    num_control_points=4,
    num_points=25,
    max_iterations=10,
    tolerance=1e-3
)

run_lengths = final_runs_df.groupby("run_id").size()
print(run_lengths.describe())
print("Runs with at least 25 frames:", np.sum(run_lengths >= 25))

# 1. Filter valid runs first
traj_tensor, valid_run_ids_ae = get_valid_resampled_tensor(final_runs_df, num_points=25, return_indices=True)

print(f"Valid runs for Autoencoder clustering: {len(valid_run_ids_ae)}")
print("Valid run IDs sample:", valid_run_ids_ae[:5])

valid_runs_df_ae = final_runs_df[final_runs_df["run_id"].isin(valid_run_ids_ae)].copy()

print("final_runs_df['run_id'] sample values:", final_runs_df["run_id"].unique()[:5])
print("final_runs_df['run_id'] dtype:", final_runs_df["run_id"].dtype)
print("Sample type of run_id in final_runs_df:", type(final_runs_df["run_id"].iloc[0]) if len(final_runs_df) > 0 else "Empty")

print("valid_run_ids_ae sample values:", valid_run_ids_ae[:5])
print("Type of elements in valid_run_ids_ae:", type(valid_run_ids_ae[0]) if len(valid_run_ids_ae) > 0 else "Empty")
print("Are all run_ids integers?:", all(isinstance(rid, (int, np.integer)) for rid in valid_run_ids_ae))

autoencoder_assignments_df, model = run_autoencoder_clustering(valid_runs_df_ae)



# Prepare tensor used in AE training and evaluation
# traj_tensor = []
# run_ids = autoencoder_assignments_df["run_id"].values
# for run_id in run_ids:
#     run_df = final_runs_df[final_runs_df["run_id"] == run_id]
#     traj = run_df[["x_mirror_c", "y_mirror_c"]].values
#     traj_tensor.append(traj)
# traj_tensor = np.stack(traj_tensor, axis=0)  # [N, 50, 2]

# Infer embeddings from encoder
model.eval()
with torch.no_grad():
    input_tensor = torch.tensor(traj_tensor, dtype=torch.float32).to(next(model.parameters()).device)
    all_z = model.encoder(input_tensor).cpu().numpy()

# For Bézier clustering
bezier_merged_df = merge_assignments_with_metadata(bezier_assignments_df, final_runs_df, cluster_col="assigned_cluster")
bezier_merged_df = add_position_buckets(bezier_merged_df)
bezier_merged_df = bezier_merged_df.merge(zones_df.drop(columns=["position", "team_role"], errors="ignore"), on="run_id", how="left")
bezier_position_pivot = compute_position_pivot(bezier_merged_df, cluster_col="assigned_cluster")
bezier_bucket_pivot = compute_bucket_pivot(bezier_merged_df, cluster_col="assigned_cluster")
assignments_zones_bezier = bezier_merged_df.copy()
cluster_control_points_bezier = bezier_cluster_control_points

# For Autoencoder clustering
ae_merged_df = merge_assignments_with_metadata(autoencoder_assignments_df, final_runs_df, cluster_col="ae_cluster")
ae_merged_df = ae_merged_df.rename(columns={"ae_cluster": "assigned_cluster"})
ae_merged_df = add_position_buckets(ae_merged_df)
ae_merged_df = ae_merged_df.merge(zones_df.drop(columns=["position", "team_role"], errors="ignore"), on="run_id", how="left")
ae_position_pivot = compute_position_pivot(ae_merged_df, cluster_col="assigned_cluster")
ae_bucket_pivot = compute_bucket_pivot(ae_merged_df, cluster_col="assigned_cluster")
assignments_zones_ae = ae_merged_df.copy()
k_clusters = 70
cluster_control_points_ae = [None] * k_clusters 

# Save Bézier clustering data
final_runs_df.to_parquet("outputs/final_runs_df.parquet", index=False)
assignments_zones_bezier.to_pickle("outputs/assignments_zones_bezier.pkl")
bezier_bucket_pivot.to_pickle("outputs/bezier_bucket_pivot.pkl")
np.save("outputs/bezier_cluster_control_points.npy", cluster_control_points_bezier)

# Save Autoencoder clustering data
assignments_zones_ae.to_parquet("outputs/assignments_zones_ae.pkl")
ae_bucket_pivot.to_parquet("outputs/ae_bucket_pivot.pkl")
np.save("outputs/autoencoder_cluster_control_points.npy", np.array(cluster_control_points_ae, dtype=object))

# ---------------------
# Transformer Clustering
# ---------------------

# 1. Filter valid runs first
final_runs_tensor, valid_run_ids_transformer = get_valid_resampled_tensor(
    final_runs_df, num_points=25, return_indices=True
)
valid_runs_df_transformer = final_runs_df[
    final_runs_df["run_id"].isin(valid_run_ids_transformer)
].copy()

transformer_assignments_df, transformer_model, transformer_loss = run_transformer_clustering(valid_runs_df_transformer)

# Merge metadata
transformer_merged_df = merge_assignments_with_metadata(transformer_assignments_df, final_runs_df, cluster_col="transformer_cluster")
transformer_merged_df = transformer_merged_df.rename(columns={"transformer_cluster": "assigned_cluster"})

# Add position buckets
transformer_merged_df = add_position_buckets(transformer_merged_df)

# Merge with zone/run features
transformer_merged_df = transformer_merged_df.merge(
    zones_df.drop(columns=["position", "team_role"], errors="ignore"),
    on="run_id",
    how="left"
)

# Create summary pivot tables
transformer_position_pivot = compute_position_pivot(transformer_merged_df, cluster_col="assigned_cluster")
transformer_bucket_pivot = compute_bucket_pivot(transformer_merged_df, cluster_col="assigned_cluster")

# Store results
assignments_zones_transformer = transformer_merged_df.copy()
cluster_control_points_transformer = [None] * 70  # Placeholder for now

# Save Transformer clustering data
assignments_zones_transformer.to_pickle("outputs/assignments_zones_transformer.pkl")
transformer_bucket_pivot.to_pickle("outputs/transformer_bucket_pivot.pkl")
np.save("outputs/transformer_cluster_control_points.npy", np.array(cluster_control_points_transformer, dtype=object))

# Prepare transformer input tensor
# final_runs_tensor = []
# run_ids_t = transformer_assignments_df["run_id"].values
# for run_id in run_ids_t:
#     run_df = final_runs_df[final_runs_df["run_id"] == run_id]
#     coords = run_df[["x_mirror_c", "y_mirror_c"]].values
#     final_runs_tensor.append(coords)
# final_runs_tensor = np.stack(final_runs_tensor, axis=0)  # [N, 50, 2]



# Run transformer model to get latent vectors
transformer_model.eval()
with torch.no_grad():
    transformer_embeddings = transformer_model(torch.tensor(final_runs_tensor, dtype=torch.float32)).cpu().numpy()

print("AE embeddings:", all_z.shape)
print("AE assignment labels:", autoencoder_assignments_df["ae_cluster"].shape)
print("AE assignment_df columns:", autoencoder_assignments_df.columns)
print("AE assignment_df preview:")
print(autoencoder_assignments_df.head())

run_all_metrics(
    autoencoder_model=model,
    ae_tensor=torch.tensor(traj_tensor, dtype=torch.float32),  # shape [N, 50, 2]
    ae_embeddings=all_z,  # from AE encoder
    ae_assignments_df=autoencoder_assignments_df,
    bezier_assignments_df=assignments_zones_bezier,
    transformer_embeddings=transformer_embeddings,  # or cached
    transformer_assignments_df=transformer_assignments_df,
    bezier_control_points=cluster_control_points_bezier
)

end_time = time.time()
elapsed = end_time - start_time

print(f"\nPipeline completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

# === Cluster Size Experiment ===
cluster_sizes = [30, 40, 50, 60, 70, 80, 90, 100]
experiment_results = []

for k in cluster_sizes:
    print(f"\n\n===== Running clustering evaluation for K={k} =====")

    # Bézier Clustering
    bezier_cp, bezier_df = bezier_kmeans_clustering(
        final_runs_df=final_runs_df,
        adjusted_runs_list=adjusted_runs_list,
        k_clusters=k,
        num_control_points=4,
        num_points=25,
        max_iterations=2,
        tolerance=1e-3
    )

    # Autoencoder Clustering
    traj_tensor, valid_ids_ae = get_valid_resampled_tensor(final_runs_df, num_points=25, return_indices=True)
    valid_df_ae = final_runs_df[final_runs_df["run_id"].isin(valid_ids_ae)].copy()
    ae_assignments_df, ae_model = run_autoencoder_clustering(valid_df_ae, k_clusters=k)

    ae_model.eval()
    with torch.no_grad():
        ae_embeddings = ae_model.encoder(torch.tensor(traj_tensor, dtype=torch.float32)).cpu().numpy()

    # Transformer Clustering
    transformer_tensor, valid_ids_t = get_valid_resampled_tensor(final_runs_df, num_points=25, return_indices=True)
    valid_df_t = final_runs_df[final_runs_df["run_id"].isin(valid_ids_t)].copy()
    transformer_assignments_df, transformer_model, transformer_loss = run_transformer_clustering(valid_df_t, num_points=25, k_clusters=k)

    transformer_model.eval()
    with torch.no_grad():
        transformer_embeddings = transformer_model(torch.tensor(transformer_tensor, dtype=torch.float32)).cpu().numpy()

    # === Run metrics and collect ===
    metrics = run_all_metrics(
        autoencoder_model=ae_model,
        ae_tensor=torch.tensor(traj_tensor, dtype=torch.float32),
        ae_embeddings=ae_embeddings,
        ae_assignments_df=ae_assignments_df,
        bezier_assignments_df=bezier_df,
        transformer_embeddings=transformer_embeddings,
        transformer_assignments_df=transformer_assignments_df,
        bezier_control_points=bezier_cp
    )

    metrics["k_clusters"] = k
    experiment_results.append(metrics)

# Save metrics to CSV
results_df_clusters = pd.DataFrame(experiment_results)
results_df_clusters.to_csv("outputs/clustering_metrics_by_k.csv", index=False)
print("\nSaved experiment results to outputs/clustering_metrics_by_k.csv")

