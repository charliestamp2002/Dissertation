import pandas as pd
import numpy as np
import time

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


start_time = time.time()
# Optional: limit number of files for quick testing
# full number of files = 1127
N_FILES = 50

# Load tracking data
print("Loading tracking data...")
frames_df, players_df, used_match_ids = load_tracking_data(n_files=N_FILES, block_size=100, num_blocks=20, seed=42)
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

print(final_runs_df.head())

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
    num_points=50,
    max_iterations=2,
    tolerance=1e-3
)

autoencoder_assignments_df, model = run_autoencoder_clustering(final_runs_df)

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
transformer_assignments_df, transformer_model = run_transformer_clustering(final_runs_df)

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

end_time = time.time()
elapsed = end_time - start_time

print(f"\nPipeline completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")