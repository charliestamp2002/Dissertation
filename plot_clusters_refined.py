import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting_refined import (
    plot_hierarchical_clusters, 
    plot_cluster_comparison,
    create_cluster_tree_visualization
)

# ====================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ====================================================================
# CHECK BEZIER IS RESAMPLING CORRECTLY. SHAPES LOOK OFF. 
# === CHOOSE CLUSTERING METHOD ===
method = "transformer"  # Options: "bezier", "autoencoder", "transformer"

# === CHOOSE PLOT MODE ===
plot_mode = "original_only"  # Options:
# "original_only"     - Plot only original clusters (traditional view)
# "subclusters_only"  - Plot all refined sub-clusters separately
# "specific_original" - Plot sub-clusters of one specific original cluster only
# "all_hierarchical"  - Plot with hierarchical structure grouping

# === SPECIFIC CLUSTER SETTINGS ===
specific_original_cluster = 64 # Only used if plot_mode = "specific_original"

# === DISPLAY SETTINGS ===
max_runs_per_cluster = 200
save_plots = True  # Set to True to save instead of showing
save_directory = "outputs/plots/"
figsize_scale = 1.0  # Scale figure size (1.0 = default, 1.5 = 50% larger)

# === COMPARISON PLOTS ===
show_comparison = True  # Show side-by-side original vs refined comparison
comparison_cluster_ids = [21,22]  # Which original clusters to compare

# === TREE VISUALIZATION ===
show_tree_structure = True  # Print tree structure to console

# ====================================================================
# LOAD DATA
# ====================================================================

print(f"Loading refined clustering data for method: {method}")

# Load base data
final_runs_df = pd.read_parquet("outputs/final_runs_df.parquet")

# Load refined clustering results
if method == "bezier":
    refined_assignments = pd.read_pickle("outputs/bezier_refined_assignments.pkl")
    cluster_analysis = pd.read_csv("outputs/bezier_cluster_analysis.csv")
    
    # Load tactical labels
    tactical_labels = {}
    try:
        with open("outputs/bezier_tactical_labels.txt", "r") as f:
            for line in f:
                if ':' in line:
                    cluster_id, label = line.strip().split(':', 1)
                    tactical_labels[cluster_id.strip()] = label.strip()
    except FileNotFoundError:
        print("Warning: Tactical labels file not found")
        tactical_labels = {}
    
    # For comparison with original clusters
    original_assignments = pd.read_pickle("outputs/assignments_zones_bezier.pkl")
    title = "Bézier"
    is_autoencoder = False

elif method == "autoencoder":
    refined_assignments = pd.read_pickle("outputs/autoencoder_refined_assignments.pkl")
    cluster_analysis = pd.read_csv("outputs/autoencoder_cluster_analysis.csv")
    
    # Load tactical labels
    tactical_labels = {}
    try:
        with open("outputs/autoencoder_tactical_labels.txt", "r") as f:
            for line in f:
                if ':' in line:
                    cluster_id, label = line.strip().split(':', 1)
                    tactical_labels[cluster_id.strip()] = label.strip()
    except FileNotFoundError:
        print("Warning: Tactical labels file not found")
        tactical_labels = {}
    
    # For comparison with original clusters
    original_assignments = pd.read_pickle("outputs/assignments_zones_ae.pkl")
    title = "Autoencoder"
    is_autoencoder = True

elif method == "transformer":
    refined_assignments = pd.read_pickle("outputs/transformer_refined_assignments.pkl")
    cluster_analysis = pd.read_csv("outputs/transformer_cluster_analysis.csv")
    
    # Load tactical labels
    tactical_labels = {}
    try:
        with open("outputs/transformer_tactical_labels.txt", "r") as f:
            for line in f:
                if ':' in line:
                    cluster_id, label = line.strip().split(':', 1)
                    tactical_labels[cluster_id.strip()] = label.strip()
    except FileNotFoundError:
        print("Warning: Tactical labels file not found")
        tactical_labels = {}
    
    # For comparison with original clusters
    original_assignments = pd.read_pickle("outputs/assignments_zones_transformer.pkl")
    title = "Transformer"
    is_autoencoder = True

else:
    raise ValueError(f"Unsupported method: {method}. Choose from: bezier, autoencoder, transformer")

print(f"Loaded {len(refined_assignments)} run assignments")
print(f"Found {len(cluster_analysis)} refined clusters")
print(f"Loaded {len(tactical_labels)} tactical labels")

# ====================================================================
# SHOW TREE STRUCTURE
# ====================================================================

if show_tree_structure:
    create_cluster_tree_visualization(refined_assignments, cluster_analysis, tactical_labels)

# ====================================================================
# MAIN HIERARCHICAL PLOT
# ====================================================================

print(f"\nGenerating {plot_mode} plot...")

fig = plot_hierarchical_clusters(
    final_runs_df=final_runs_df,
    refined_assignments=refined_assignments,
    cluster_analysis=cluster_analysis,
    tactical_labels=tactical_labels,
    plot_mode=plot_mode,
    specific_original_cluster=specific_original_cluster,
    max_runs_per_cluster=max_runs_per_cluster,
    title=title,
    is_autoencoder=is_autoencoder,
    plot_absolute_positions=False,
    figsize_scale=figsize_scale,
    # Add any additional filters here if needed:
    # === ZONE FILTERS ===
    #start_zones=[1,2,3],  # Filter by starting zone (1-6)
    # end_zones=[4, 5, 6],    # Filter by ending zone (1-6)
    # use_absolute_zones=False,  # Use absolute zones instead of relative
    # start_zones_absolute=[1, 2, 3],  # Absolute start zones (1-6)
    # end_zones_absolute=[4, 5, 6],    # Absolute end zones (1-6)
    
    # === PHASE OF PLAY FILTERS ===
    # phases_of_play=["attack"],  # Options: "attack", "defense", "transition"
    
    # === POSITION FILTERS ===
    # positions=["CM", "CAM", "RW"],  # Filter by player positions
    
    # === RUN CHARACTERISTICS ===
    # run_angle_range=(-45, 45),  # Angle range in degrees
    # run_forward=True,           # True for forward runs, False for backward
    #run_length_range=(5.0, 30.0),  # Run length in meters (min, max)
    # mean_speed_range=(2.0, 8.0),   # Mean speed in m/s (min, max)
    # max_speed_range=(5.0, 12.0),   # Max speed in m/s (min, max)
    
    # === TACTICAL CONTEXT FILTERS ===
    # tactical_overlap=True,      # Runs with overlapping movement
    #tactical_underlap=False,    # Runs with underlapping movement  
    # tactical_diagonal=True,     # Diagonal runs
    #runner_received_pass=True,  # Runs where runner received a pass
)

if save_plots:
    import os
    os.makedirs(save_directory, exist_ok=True)
    filename = f"{method}_{plot_mode}"
    if plot_mode == "specific_original":
        filename += f"_cluster_{specific_original_cluster}"
    filename += ".pdf"
    
    plt.figure(fig.number)
    plt.savefig(f"{save_directory}{filename}", dpi=300, bbox_inches="tight")
    print(f"Main plot saved to: {save_directory}{filename}")
else:
    plt.show()

# ====================================================================
# COMPARISON PLOTS
# ====================================================================

if show_comparison and plot_mode != "original_only":
    print(f"\nGenerating comparison plots for clusters: {comparison_cluster_ids}")
    
    for cluster_id in comparison_cluster_ids:
        # Check if this cluster exists and was refined
        cluster_exists = str(cluster_id) in refined_assignments['base_cluster'].values
        if not cluster_exists:
            print(f"Warning: Cluster {cluster_id} not found in data, skipping...")
            continue
            
        # Check if cluster was actually refined (has multiple sub-clusters)
        subclusters = refined_assignments[
            refined_assignments['base_cluster'] == str(cluster_id)
        ]['refined_cluster'].unique()
        
        if len(subclusters) <= 1:
            print(f"Cluster {cluster_id} was not refined (only 1 sub-cluster), skipping comparison...")
            continue
        
        print(f"Creating comparison for original cluster {cluster_id} -> {len(subclusters)} sub-clusters")
        
        comparison_fig = plot_cluster_comparison(
            final_runs_df=final_runs_df,
            original_assignments=original_assignments,
            refined_assignments=refined_assignments,
            cluster_analysis=cluster_analysis,
            tactical_labels=tactical_labels,
            original_cluster_id=cluster_id,
            max_runs_per_cluster=20,
            title=f"{title} Cluster {cluster_id} Refinement"
        )
        
        if save_plots:
            filename = f"{method}_comparison_cluster_{cluster_id}.png"
            plt.figure(comparison_fig.number)
            plt.savefig(f"{save_directory}{filename}", dpi=300, bbox_inches="tight")
            print(f"Comparison plot saved to: {save_directory}{filename}")
        else:
            plt.show()

# ====================================================================
# SUMMARY STATISTICS
# ====================================================================

print(f"\n" + "="*60)
print(f"REFINED CLUSTERING SUMMARY - {method.upper()}")
print("="*60)

# Original cluster statistics
original_clusters = len(refined_assignments['base_cluster'].unique())
refined_clusters = len(refined_assignments['refined_cluster'].unique())
total_runs = len(refined_assignments)

print(f"Original clusters: {original_clusters}")
print(f"Refined clusters: {refined_clusters}")
print(f"Total runs: {total_runs}")
print(f"Average runs per refined cluster: {total_runs/refined_clusters:.1f}")

# Refinement statistics
refinement_stats = refined_assignments.groupby('base_cluster').agg({
    'refined_cluster': 'nunique',
    'run_id': 'count'
}).rename(columns={'refined_cluster': 'n_subclusters', 'run_id': 'n_runs'})

refined_count = len(refinement_stats[refinement_stats['n_subclusters'] > 1])
print(f"Clusters that were refined: {refined_count}/{original_clusters} ({100*refined_count/original_clusters:.1f}%)")

# Top tactical patterns
print(f"\nTop 10 Most Common Tactical Patterns:")
tactical_counts = {}
for cluster_id, label in tactical_labels.items():
    n_runs = len(refined_assignments[refined_assignments['refined_cluster'] == cluster_id])
    if label in tactical_counts:
        tactical_counts[label] += n_runs
    else:
        tactical_counts[label] = n_runs

sorted_patterns = sorted(tactical_counts.items(), key=lambda x: x[1], reverse=True)
for i, (pattern, count) in enumerate(sorted_patterns[:10]):
    print(f"{i+1:2d}. {pattern}: {count} runs")

print("="*60)

# ====================================================================
# INTERACTIVE OPTIONS
# ====================================================================

print(f"\nInteractive Options:")
print(f"1. To plot a specific original cluster's sub-clusters:")
print(f"   Set plot_mode = 'specific_original' and specific_original_cluster = <cluster_id>")
print(f"")
print(f"2. To add filters (e.g., only attacking runs):")
print(f"   Add parameters like phases_of_play=['attack'] to plot_hierarchical_clusters()")
print(f"")
print(f"3. To save plots:")
print(f"   Set save_plots = True")
print(f"")
print(f"4. Available original clusters to explore: {sorted([int(x) for x in refined_assignments['base_cluster'].unique() if str(x).isdigit()])}")

# Show available cluster IDs and their refinement status
print(f"\nCluster Refinement Status:")
for base_cluster in sorted(refinement_stats.index, key=lambda x: int(x) if str(x).isdigit() else float('inf')):
    n_subs = refinement_stats.loc[base_cluster, 'n_subclusters']
    n_runs = refinement_stats.loc[base_cluster, 'n_runs']
    status = "REFINED" if n_subs > 1 else "single"
    print(f"  Cluster {base_cluster}: {n_runs} runs -> {n_subs} sub-cluster{'s' if n_subs > 1 else ''} [{status}]")

print(f"\nDone!")