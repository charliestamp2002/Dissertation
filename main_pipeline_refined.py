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

# NEW: Tactical refinement imports
from tactical_refinement import (
    tactical_refinement, 
    analyze_refined_clusters, 
    create_tactical_cluster_labels
)

start_time = time.time()
# Optional: limit number of files for quick testing
# full number of files = 1127
# Quarter season = 95 games (380 total matches / 4)
N_FILES = 2

# Load tracking data
print("Loading tracking data...")
frames_df, players_df, used_match_ids = load_tracking_data(n_files=N_FILES, block_size=500, num_blocks=10, seed=42)
print(f"Loaded {len(frames_df)} frames and {len(players_df)} player entries from {len(used_match_ids)} matches.\n")

# Load event metadata
print("Loading StatsBomb event metadata...")
event_meta_df = load_event_metadata(used_match_ids=used_match_ids)
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

# Align event frames
print("Aligning event frames with tracking data...")
events_dfs_by_match = load_and_align_events(frames_df, players_df, event_tracking_df_clean)
print(f"Aligned events for {len(events_dfs_by_match)} matches.")

# Preview one match
debug_match_id = "tracking_" + event_tracking_df_clean.iloc[0]["tracking_suffix"]

# Attach player metadata
print("\nBuilding player metadata from Opta data...")
players_df, metadata_df, metadata_dict = build_opta_metadata(players_df, used_match_ids)
print(f"Built metadata for {len(metadata_dict)} players.")

# 1. Segment runs from player data
print("\nSegmenting player runs from tracking data...")
runs_list = segment_runs(players_df)
print(f"Segmented {len(runs_list)} total runs.")

# 2. Filter to off-ball runs
print("Filtering to off-ball runs (min distance: 3.0m)...")
off_ball_runs = filter_off_ball_runs_with_distance(runs_list, frames_df, min_distance=3.0)
print(f"Filtered to {len(off_ball_runs)} off-ball runs.")

# 3. Annotate and normalize runs
print("Processing and normalizing runs...")
adjusted_runs_list ,final_runs_df = process_runs(players_df, frames_df, off_ball_runs)
print(f"Processed {len(final_runs_df)} run data points.")

run_lengths = final_runs_df.groupby("run_id").size()

print("\nExtracting zone and run features with enhanced tactical context...")
zones_df = extract_zone_and_run_features(
    final_runs_df,
    players_df,
    frames_df,
    events_dfs_by_match,
    metadata_dict
)
print(f"Extracted features for {len(zones_df)} runs.")

print("\nTactical features sample:")
tactical_cols = ['build_up_phase', 'counter_attack', 'attacking_third_entry', 
                'creates_passing_lane', 'stretches_defense', 'tactical_overlap']
available_cols = [col for col in tactical_cols if col in zones_df.columns]
if available_cols:
    print(zones_df[available_cols].head())
else:
    print("No tactical features found in zones_df")

# 1. Filter valid runs first
traj_tensor, valid_run_ids_ae = get_valid_resampled_tensor(final_runs_df, num_points=25, return_indices=True)

print(f"\nValid runs for clustering: {len(valid_run_ids_ae)}")

valid_runs_df_ae = final_runs_df[final_runs_df["run_id"].isin(valid_run_ids_ae)].copy()

print("\n=== STARTING BEZIER CLUSTERING ===")
print("Running Bézier K-means clustering (k=70, 4 control points, 25 sample points)...")

# Convert Series to list if needed
if isinstance(valid_run_ids_ae, pd.Series):
    valid_run_ids_ae = valid_run_ids_ae.tolist()
valid_run_ids_ae_set = set(valid_run_ids_ae)

# Fix: Extract run_id from each run DataFrame (assume all rows in a run have same run_id)
adjusted_valid_runs_list = []
for run in adjusted_runs_list:
    if isinstance(run, pd.DataFrame) and "run_id" in run.columns:
        run_id = run["run_id"].iloc[0]  # scalar value
        if run_id in valid_run_ids_ae_set:
            adjusted_valid_runs_list.append(run)

bezier_cluster_control_points, bezier_assignments_df = bezier_kmeans_clustering(
    final_runs_df=final_runs_df,
    adjusted_runs_list=adjusted_valid_runs_list,
    k_clusters=70,
    num_control_points=4,
    num_points=25,
    max_iterations=20,
    tolerance=1e-3
)
print(f"Bézier clustering completed. Assigned {len(bezier_assignments_df)} runs to clusters.")

print("\n=== STARTING AUTOENCODER CLUSTERING ===")
print("Running autoencoder clustering...")
autoencoder_assignments_df, model = run_autoencoder_clustering(valid_runs_df_ae)
print(f"Autoencoder clustering completed. Assigned {len(autoencoder_assignments_df)} runs to clusters.")

# Infer embeddings from encoder
print("Generating embeddings from trained autoencoder...")
model.eval()
with torch.no_grad():
    input_tensor = torch.tensor(traj_tensor, dtype=torch.float32).to(next(model.parameters()).device)
    all_z = model.encoder(input_tensor).cpu().numpy()
print(f"Generated embeddings of shape: {all_z.shape}")

print("\n=== STARTING TRANSFORMER CLUSTERING ===")
print("Filtering valid runs for transformer clustering...")
# 1. Filter valid runs first
final_runs_tensor, valid_run_ids_transformer = get_valid_resampled_tensor(
    final_runs_df, num_points=25, return_indices=True
)
valid_runs_df_transformer = final_runs_df[
    final_runs_df["run_id"].isin(valid_run_ids_transformer)
].copy()

print(f"Valid runs for transformer clustering: {len(valid_runs_df_transformer)}")
print("Running transformer clustering...")
transformer_assignments_df, transformer_model, transformer_loss = run_transformer_clustering(valid_runs_df_transformer)
print(f"Transformer clustering completed. Assigned {len(transformer_assignments_df)} runs to clusters.")

# Run transformer model to get latent vectors
print("Generating embeddings from trained transformer...")
transformer_model.eval()
with torch.no_grad():
    input_tensor = torch.tensor(final_runs_tensor, dtype=torch.float32).to(next(transformer_model.parameters()).device)
    transformer_embeddings = transformer_model(input_tensor).cpu().numpy()
    print(f"Generated transformer embeddings of shape: {transformer_embeddings.shape}")

print("\n" + "="*60)
print("=== TACTICAL REFINEMENT PHASE ===")
print("="*60)

# Apply tactical refinement to each clustering method
print("\n1. Refining Bézier clusters with tactical context...")
bezier_refined_df, bezier_stats = tactical_refinement(
    bezier_assignments_df, zones_df, min_cluster_size=8, max_subclusters=4
)

print("\n2. Refining Autoencoder clusters with tactical context...")
autoencoder_refined_df, ae_stats = tactical_refinement(
    autoencoder_assignments_df, zones_df, min_cluster_size=8, max_subclusters=4
)

print("\n3. Refining Transformer clusters with tactical context...")
transformer_refined_df, transformer_stats = tactical_refinement(
    transformer_assignments_df, zones_df, min_cluster_size=8, max_subclusters=4
)

# Analyze refined clusters
print("\n=== ANALYZING REFINED CLUSTERS ===")

print("\nAnalyzing Bézier refined clusters...")
bezier_cluster_analysis = analyze_refined_clusters(bezier_refined_df, zones_df)
bezier_tactical_labels = create_tactical_cluster_labels(bezier_cluster_analysis)

print("\nAnalyzing Autoencoder refined clusters...")
ae_cluster_analysis = analyze_refined_clusters(autoencoder_refined_df, zones_df)
ae_tactical_labels = create_tactical_cluster_labels(ae_cluster_analysis)

print("\nAnalyzing Transformer refined clusters...")
transformer_cluster_analysis = analyze_refined_clusters(transformer_refined_df, zones_df)
transformer_tactical_labels = create_tactical_cluster_labels(transformer_cluster_analysis)

# Merge refined assignments with metadata for final results
print("\n=== CREATING FINAL REFINED RESULTS ===")

# For Bézier clustering
print("Merging Bézier refined clustering results with metadata...")
bezier_refined_merged = bezier_refined_df.merge(
    merge_assignments_with_metadata(bezier_assignments_df, final_runs_df, cluster_col="assigned_cluster"),
    on="run_id", how="left"
)
bezier_refined_merged = add_position_buckets(bezier_refined_merged)
bezier_refined_merged = bezier_refined_merged.merge(
    zones_df.drop(columns=["position", "team_role"], errors="ignore"), 
    on="run_id", how="left"
)

# Add tactical labels
bezier_refined_merged['tactical_label'] = bezier_refined_merged['refined_cluster'].map(bezier_tactical_labels)

# For Autoencoder clustering
print("Merging Autoencoder refined clustering results with metadata...")
ae_refined_merged = autoencoder_refined_df.merge(
    merge_assignments_with_metadata(autoencoder_assignments_df, final_runs_df, cluster_col="ae_cluster"),
    on="run_id", how="left"
)
ae_refined_merged = add_position_buckets(ae_refined_merged)
ae_refined_merged = ae_refined_merged.merge(
    zones_df.drop(columns=["position", "team_role"], errors="ignore"), 
    on="run_id", how="left"
)
ae_refined_merged['tactical_label'] = ae_refined_merged['refined_cluster'].map(ae_tactical_labels)

# For Transformer clustering
print("Merging Transformer refined clustering results with metadata...")
transformer_refined_merged = transformer_refined_df.merge(
    merge_assignments_with_metadata(transformer_assignments_df, final_runs_df, cluster_col="transformer_cluster"),
    on="run_id", how="left"
)
transformer_refined_merged = add_position_buckets(transformer_refined_merged)
transformer_refined_merged = transformer_refined_merged.merge(
    zones_df.drop(columns=["position", "team_role"], errors="ignore"), 
    on="run_id", how="left"
)
transformer_refined_merged['tactical_label'] = transformer_refined_merged['refined_cluster'].map(transformer_tactical_labels)

print("\n=== SAVING REFINED RESULTS ===")

# Save base data needed for plotting
print("Saving base data...")
final_runs_df.to_parquet("outputs/final_runs_df.parquet", index=False)
zones_df.to_pickle("outputs/zones_df.pkl")

# Save original clustering results (needed for comparisons)
print("Saving original clustering results...")
bezier_assignments_df.to_pickle("outputs/assignments_zones_bezier.pkl") 
autoencoder_assignments_df.to_pickle("outputs/assignments_zones_ae.pkl")
transformer_assignments_df.to_pickle("outputs/assignments_zones_transformer.pkl")

# Save all refined results
print("Saving Bézier refined clustering results...")
bezier_refined_merged.to_pickle("outputs/bezier_refined_assignments.pkl")
bezier_cluster_analysis.to_csv("outputs/bezier_cluster_analysis.csv", index=False)
bezier_stats.to_csv("outputs/bezier_refinement_stats.csv", index=False)

print("Saving Autoencoder refined clustering results...")
ae_refined_merged.to_pickle("outputs/autoencoder_refined_assignments.pkl")
ae_cluster_analysis.to_csv("outputs/autoencoder_cluster_analysis.csv", index=False)
ae_stats.to_csv("outputs/autoencoder_refinement_stats.csv", index=False)

print("Saving Transformer refined clustering results...")
transformer_refined_merged.to_pickle("outputs/transformer_refined_assignments.pkl")
transformer_cluster_analysis.to_csv("outputs/transformer_cluster_analysis.csv", index=False)
transformer_stats.to_csv("outputs/transformer_refinement_stats.csv", index=False)

# Save tactical labels
with open("outputs/bezier_tactical_labels.txt", "w") as f:
    for cluster_id, label in bezier_tactical_labels.items():
        f.write(f"{cluster_id}: {label}\n")

with open("outputs/autoencoder_tactical_labels.txt", "w") as f:
    for cluster_id, label in ae_tactical_labels.items():
        f.write(f"{cluster_id}: {label}\n")

with open("outputs/transformer_tactical_labels.txt", "w") as f:
    for cluster_id, label in transformer_tactical_labels.items():
        f.write(f"{cluster_id}: {label}\n")

print("\n=== REFINED CLUSTERING SUMMARY ===")

print(f"\nBézier Clustering:")
print(f"  Original clusters: {bezier_assignments_df['assigned_cluster'].nunique()}")
print(f"  Refined clusters: {bezier_refined_df['refined_cluster'].nunique()}")
print(f"  Clusters refined: {bezier_stats['refinement_applied'].sum()}")

print(f"\nAutoencoder Clustering:")
print(f"  Original clusters: {autoencoder_assignments_df['ae_cluster'].nunique()}")
print(f"  Refined clusters: {autoencoder_refined_df['refined_cluster'].nunique()}")
print(f"  Clusters refined: {ae_stats['refinement_applied'].sum()}")

print(f"\nTransformer Clustering:")
print(f"  Original clusters: {transformer_assignments_df['transformer_cluster'].nunique()}")
print(f"  Refined clusters: {transformer_refined_df['refined_cluster'].nunique()}")  
print(f"  Clusters refined: {transformer_stats['refinement_applied'].sum()}")

# Show some example tactical labels
print(f"\nExample Bézier Tactical Labels:")
for i, (cluster_id, label) in enumerate(list(bezier_tactical_labels.items())[:5]):
    n_runs = len(bezier_refined_df[bezier_refined_df['refined_cluster'] == cluster_id])
    print(f"  {cluster_id}: {label} ({n_runs} runs)")

# Standard metrics on original (non-refined) clusters
print("\n=== RUNNING EVALUATION METRICS ON ORIGINAL CLUSTERS ===")
print("Computing clustering evaluation metrics for all methods...")

run_all_metrics(
    autoencoder_model=model,
    ae_tensor=torch.tensor(traj_tensor, dtype=torch.float32),  # shape [N, 50, 2]
    ae_embeddings=all_z,  # from AE encoder
    ae_assignments_df=autoencoder_assignments_df,
    bezier_assignments_df=bezier_assignments_df,
    transformer_embeddings=transformer_embeddings,
    transformer_assignments_df=transformer_assignments_df,
    bezier_control_points=bezier_cluster_control_points
)
print("Metrics computation completed.")

end_time = time.time()
elapsed = end_time - start_time

print(f"\n" + "="*60)
print(f"REFINED PIPELINE COMPLETED IN {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print("="*60)

print("\nKey outputs saved to outputs/ directory:")
print("  - *_refined_assignments.pkl: Final cluster assignments with tactical context")
print("  - *_cluster_analysis.csv: Detailed tactical analysis of each cluster")
print("  - *_refinement_stats.csv: Statistics on the refinement process")
print("  - *_tactical_labels.txt: Human-readable tactical labels for clusters")