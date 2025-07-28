import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from matplotlib.patches import FancyBboxPatch, Circle
from bezier_clustering import resample_coords, fit_bezier_curve, evaluate_bezier_curve
import pandas as pd


def draw_pitch(ax, pitch_length=105, pitch_width=68):
    half_length = pitch_length / 2
    half_width = pitch_width / 2

    ax.plot([-half_length, -half_length, half_length, half_length, -half_length],
            [-half_width, half_width, half_width, -half_width, -half_width], color="black")
    ax.plot([0, 0], [-half_width, half_width], color="black")

    ax.plot([-half_length + 16.5, -half_length + 16.5], [-13.84, 13.84], color="black")
    ax.plot([-half_length, -half_length + 16.5], [-13.84, -13.84], color="black")
    ax.plot([-half_length, -half_length + 16.5], [13.84, 13.84], color="black")

    ax.plot([half_length - 16.5, half_length - 16.5], [-13.84, 13.84], color="black")
    ax.plot([half_length, half_length - 16.5], [-13.84, -13.84], color="black")
    ax.plot([half_length, half_length - 16.5], [13.84, 13.84], color="black")

    circle = plt.Circle((0, 0), 9.15, color="black", fill=False)
    ax.add_patch(circle)

    ax.set_xlim(-half_length, half_length)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect("equal")
    ax.axis("off")


def get_cluster_colors(n_colors, base_color=None):
    """Generate distinct colors for sub-clusters"""
    if base_color is None:
        # Use matplotlib color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        return colors[:n_colors] if n_colors <= len(colors) else colors * (n_colors // len(colors) + 1)
    else:
        # Generate shades of a base color
        import matplotlib.colors as mcolors
        base_rgb = mcolors.to_rgb(base_color)
        colors = []
        for i in range(n_colors):
            alpha = 0.3 + 0.7 * i / max(1, n_colors - 1)  # From light to dark
            colors.append((*base_rgb, alpha))
        return colors


def plot_hierarchical_clusters(
    final_runs_df,
    refined_assignments,
    cluster_analysis,
    tactical_labels,
    plot_mode="original_only",  # "original_only", "subclusters_only", "specific_original", "all_hierarchical"
    specific_original_cluster=None,
    max_runs_per_cluster=30,
    title=None,
    is_autoencoder=False,
    plot_absolute_positions=True,
    figsize_scale=1.0,
    # All the existing filter parameters
    start_zones=None,
    end_zones=None,
    phases_of_play=None,
    positions=None,
    use_absolute_zones=False,
    start_zones_absolute=None,
    end_zones_absolute=None,
    run_angle_range=None,
    run_forward=None,
    run_length_range=None,
    mean_speed_range=None,
    max_speed_range=None,
    tactical_overlap=None,
    tactical_underlap=None,
    tactical_diagonal=None,
    runner_received_pass=None,
):
    """
    Enhanced plotting function for hierarchical clusters
    
    Args:
        plot_mode: 
            - "original_only": Plot only original clusters (traditional view)
            - "subclusters_only": Plot all refined sub-clusters
            - "specific_original": Plot sub-clusters of one specific original cluster
            - "all_hierarchical": Plot with hierarchical grouping
        specific_original_cluster: Which original cluster to focus on (for "specific_original" mode)
    """
    
    # Apply all filters to refined_assignments
    filtered_assignments = refined_assignments.copy()
    
    # Merge with cluster analysis to get tactical features if needed
    if 'tactical_overlap' not in filtered_assignments.columns:
        # Need to merge with zones data for filtering - assume it's in refined_assignments already
        pass
    
    # Apply filters (same as original function)
    if start_zones is not None and not use_absolute_zones:
        filtered_assignments = filtered_assignments[
            filtered_assignments["start_zone"].isin(start_zones)]
    if end_zones is not None and not use_absolute_zones:
        filtered_assignments = filtered_assignments[
            filtered_assignments["end_zone"].isin(end_zones)]
    if start_zones_absolute is not None and use_absolute_zones:
        filtered_assignments = filtered_assignments[
            filtered_assignments["start_zone_absolute"].isin(start_zones_absolute)]
    if end_zones_absolute is not None and use_absolute_zones:
        filtered_assignments = filtered_assignments[
            filtered_assignments["end_zone_absolute"].isin(end_zones_absolute)]
    if phases_of_play is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["phase_of_play"].isin(phases_of_play)]
    if positions is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["position"].isin(positions)]
    if run_angle_range is not None:
        min_angle, max_angle = run_angle_range
        filtered_assignments = filtered_assignments[
            (filtered_assignments["run_angle_deg"] >= min_angle) &
            (filtered_assignments["run_angle_deg"] <= max_angle)]
    if run_forward is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["run_forward"] == run_forward]
    if run_length_range is not None:
        min_len, max_len = run_length_range
        filtered_assignments = filtered_assignments[
            (filtered_assignments["run_length_m"] >= min_len) &
            (filtered_assignments["run_length_m"] <= max_len)]
    if mean_speed_range is not None:
        min_speed, max_speed = mean_speed_range
        filtered_assignments = filtered_assignments[
            (filtered_assignments["mean_speed"] >= min_speed) &
            (filtered_assignments["mean_speed"] <= max_speed)]
    if max_speed_range is not None:
        min_speed, max_speed = max_speed_range
        filtered_assignments = filtered_assignments[
            (filtered_assignments["max_speed"] >= min_speed) &
            (filtered_assignments["max_speed"] <= max_speed)]
    if tactical_overlap is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["tactical_overlap"] == tactical_overlap]
    if tactical_underlap is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["tactical_underlap"] == tactical_underlap]
    if tactical_diagonal is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["tactical_diagonal"] == tactical_diagonal]
    if runner_received_pass is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["runner_received_pass"] == runner_received_pass]

    # Determine what to plot based on mode
    if plot_mode == "original_only":
        # Plot using base_cluster, group sub-clusters together
        clusters_to_plot = sorted(filtered_assignments['base_cluster'].astype(str).unique(), 
                                 key=lambda x: int(x) if x.isdigit() else float('inf'))
        cluster_id_col = 'base_cluster'
        plot_title = f"Original {title} Clusters"
        
    elif plot_mode == "subclusters_only":
        # Plot all refined sub-clusters separately
        clusters_to_plot = sorted(filtered_assignments['refined_cluster'].unique())
        cluster_id_col = 'refined_cluster'
        plot_title = f"Refined {title} Sub-Clusters"
        
    elif plot_mode == "specific_original":
        # Plot only sub-clusters of a specific original cluster
        if specific_original_cluster is None:
            raise ValueError("specific_original_cluster must be specified for 'specific_original' mode")
        
        filtered_assignments = filtered_assignments[
            filtered_assignments['base_cluster'].astype(str) == str(specific_original_cluster)
        ]
        clusters_to_plot = sorted(filtered_assignments['refined_cluster'].unique())
        cluster_id_col = 'refined_cluster'
        plot_title = f"{title} Sub-Clusters of Original Cluster {specific_original_cluster}"
        
    elif plot_mode == "all_hierarchical":
        # Plot with hierarchical structure - group sub-clusters under parents
        clusters_to_plot = sorted(filtered_assignments['refined_cluster'].unique())
        cluster_id_col = 'refined_cluster'
        plot_title = f"Hierarchical {title} Clusters"
        
    else:
        raise ValueError(f"Unknown plot_mode: {plot_mode}")

    # Calculate grid size
    num_clusters = len(clusters_to_plot)
    if num_clusters == 0:
        print("No clusters to plot after filtering")
        return
        
    cols = min(10, num_clusters)
    rows = (num_clusters + cols - 1) // cols
    
    # Adjust figure size
    base_figsize = (3 * cols, 2.5 * rows)
    figsize = (base_figsize[0] * figsize_scale, base_figsize[1] * figsize_scale)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_clusters == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    # Plot each cluster
    for idx, cluster_id in enumerate(clusters_to_plot):
        ax = axes[idx]
        draw_pitch(ax)

        # Get runs for this cluster
        cluster_runs = filtered_assignments[
            filtered_assignments[cluster_id_col].astype(str) == str(cluster_id)
        ]
        cluster_run_ids = cluster_runs["run_id"].tolist()

        # Sample runs if too many
        if len(cluster_run_ids) > max_runs_per_cluster:
            cluster_run_ids = random.sample(cluster_run_ids, max_runs_per_cluster)

        # Determine color strategy
        if plot_mode == "specific_original":
            # Use different colors for each sub-cluster
            unique_subclusters = sorted(filtered_assignments['refined_cluster'].unique())
            colors = get_cluster_colors(len(unique_subclusters))
            color_idx = unique_subclusters.index(cluster_id)
            color = colors[color_idx]
        else:
            color = "blue"

        # Plot trajectories
        # print(f"\n=== DEBUG: Cluster {cluster_id} ===")
        # print(f"Number of runs to plot: {len(cluster_run_ids)}")
        
        for i, run_id in enumerate(cluster_run_ids[:3]):  # Only debug first 3 runs
            run_df = final_runs_df[final_runs_df["run_id"] == run_id]
            if len(run_df) == 0:
                print(f"Run {run_id}: No data found")
                continue
                
            coords = run_df[["x_mirror_c", "y_mirror_c"]].values
            if coords.shape[0] < 2:
                print(f"Run {run_id}: Too few points ({coords.shape[0]})")
                continue

            # print(f"Run {run_id}: Original coords shape: {coords.shape}")
            # print(f"  X range: {coords[:, 0].min():.2f} to {coords[:, 0].max():.2f}")
            # print(f"  Y range: {coords[:, 1].min():.2f} to {coords[:, 1].max():.2f}")
            # print(f"  First point: ({coords[0, 0]:.2f}, {coords[0, 1]:.2f})")
            # print(f"  Last point: ({coords[-1, 0]:.2f}, {coords[-1, 1]:.2f})")

            # Plot trajectory
            if is_autoencoder:
                trajectory = coords
                print(f"  Using autoencoder mode (direct coords)")
            else:
                resampled = resample_coords(coords, num_points=50)
                print(f"  Resampled to {len(resampled)} points")
                if len(resampled) >= 4:  # Need at least 4 points for Bézier
                    control_pts = fit_bezier_curve(resampled, num_control_points=4)
                    if plot_absolute_positions:
                        start_pos = run_df[["x", "y"]].values[0]
                        print(f"  Using absolute positions, start_pos: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
                        shifted_ctrl_pts = control_pts - control_pts[0] + start_pos
                        trajectory = evaluate_bezier_curve(shifted_ctrl_pts, num_points=50)
                    else:
                        print(f"  Using relative positions")
                        trajectory = evaluate_bezier_curve(control_pts, num_points=50)
                else:
                    trajectory = coords
                    print(f"  Using original coords (not enough points for Bézier)")
            
            # print(f"  Final trajectory shape: {trajectory.shape}")
            # print(f"  Final X range: {trajectory[:, 0].min():.2f} to {trajectory[:, 0].max():.2f}")
            # print(f"  Final Y range: {trajectory[:, 1].min():.2f} to {trajectory[:, 1].max():.2f}")

            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color=color, linewidth=1)
        
        # Continue plotting remaining runs without debug
        for run_id in cluster_run_ids[3:]:
            run_df = final_runs_df[final_runs_df["run_id"] == run_id]
            if len(run_df) == 0:
                continue
                
            coords = run_df[["x_mirror_c", "y_mirror_c"]].values
            if coords.shape[0] < 2:
                continue

            # Plot trajectory
            if is_autoencoder:
                trajectory = coords
            else:
                resampled = resample_coords(coords, num_points=50)
                if len(resampled) >= 4:  # Need at least 4 points for Bézier
                    control_pts = fit_bezier_curve(resampled, num_control_points=4)
                    if plot_absolute_positions:
                        start_pos = run_df[["x", "y"]].values[0]
                        shifted_ctrl_pts = control_pts - control_pts[0] + start_pos
                        trajectory = evaluate_bezier_curve(shifted_ctrl_pts, num_points=50)
                    else:
                        trajectory = evaluate_bezier_curve(control_pts, num_points=50)
                else:
                    trajectory = coords

            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color=color, linewidth=1)

        # Add cluster information
        n_runs = len(cluster_run_ids)
        
        # Get tactical label if available
        tactical_label = tactical_labels.get(cluster_id, "")
        if tactical_label and len(tactical_label) > 25:  # Truncate long labels
            tactical_label = tactical_label[:25] + "..."
            
        # Get cluster statistics from analysis
        cluster_stats = cluster_analysis[cluster_analysis['refined_cluster'] == cluster_id]
        stats_text = f"n={n_runs}"
        
        if len(cluster_stats) > 0:
            stats = cluster_stats.iloc[0]
            # Add key tactical percentages
            key_stats = []
            if stats.get('counter_attack_pct', 0) > 30:
                key_stats.append(f"Counter: {stats['counter_attack_pct']:.0f}%")
            if stats.get('build_up_pct', 0) > 30:
                key_stats.append(f"Build-up: {stats['build_up_pct']:.0f}%") 
            if stats.get('attacking_third_pct', 0) > 30:
                key_stats.append(f"Att.3rd: {stats['attacking_third_pct']:.0f}%")
                
            if key_stats:
                stats_text += "\n" + "\n".join(key_stats[:2])  # Max 2 stats

        # Title with hierarchical info
        if plot_mode == "subclusters_only" or plot_mode == "specific_original":
            base_cluster = cluster_id.split('_')[0]
            sub_cluster = cluster_id.split('_')[1]
            title_text = f"{base_cluster}.{sub_cluster}"
        else:
            title_text = str(cluster_id)
            
        ax.set_title(title_text, fontsize=10, fontweight='bold')
        
        # Add tactical label
        if tactical_label:
            ax.text(0.5, 1.15, tactical_label, transform=ax.transAxes, 
                   ha="center", va="bottom", fontsize=8, style='italic',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
        
        # Add statistics
        ax.text(0.5, -0.15, stats_text, transform=ax.transAxes,
               ha="center", va="top", fontsize=7,
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Hide unused subplots
    for i in range(num_clusters, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.suptitle(plot_title, fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.92)
    return fig


def plot_cluster_comparison(
    final_runs_df,
    original_assignments,
    refined_assignments, 
    cluster_analysis,
    tactical_labels,
    original_cluster_id,
    max_runs_per_cluster=20,
    title="Cluster Refinement Comparison"
):
    """
    Side-by-side comparison of original cluster vs its refined sub-clusters
    """
    
    # Get sub-clusters for the original cluster
    subclusters = refined_assignments[
        refined_assignments['base_cluster'].astype(str) == str(original_cluster_id)
    ]['refined_cluster'].unique()
    
    n_subclusters = len(subclusters)
    
    # Create figure: 1 original + n sub-clusters
    fig, axes = plt.subplots(1, n_subclusters + 1, figsize=(4 * (n_subclusters + 1), 4))
    if n_subclusters == 0:
        axes = [axes]
    
    # Plot original cluster (leftmost)
    ax = axes[0]
    draw_pitch(ax)
    
    # Get original cluster runs
    if 'assigned_cluster' in original_assignments.columns:
        cluster_col = 'assigned_cluster'
    else:
        # Try other common column names
        cluster_col = [col for col in original_assignments.columns if 'cluster' in col.lower() and col != 'run_id'][0]
    
    original_runs = original_assignments[
        original_assignments[cluster_col] == original_cluster_id
    ]
    original_run_ids = original_runs['run_id'].tolist()
    
    if len(original_run_ids) > max_runs_per_cluster:
        original_run_ids = random.sample(original_run_ids, max_runs_per_cluster)
    
    # Plot original cluster trajectories in gray
    for run_id in original_run_ids:
        run_df = final_runs_df[final_runs_df["run_id"] == run_id]
        if len(run_df) == 0:
            continue
        coords = run_df[["x_mirror_c", "y_mirror_c"]].values
        if coords.shape[0] >= 2:
            ax.plot(coords[:, 0], coords[:, 1], alpha=0.3, color="gray", linewidth=1)
    
    ax.set_title(f"Original Cluster {original_cluster_id}\n({len(original_run_ids)} runs)", 
                fontsize=10, fontweight='bold')
    
    # Plot each sub-cluster
    colors = get_cluster_colors(n_subclusters)
    
    for i, subcluster_id in enumerate(sorted(subclusters)):
        ax = axes[i + 1]
        draw_pitch(ax)
        
        # Get sub-cluster runs
        subcluster_runs = refined_assignments[
            refined_assignments['refined_cluster'].astype(str) == str(subcluster_id)
        ]
        subcluster_run_ids = subcluster_runs['run_id'].tolist()
        
        if len(subcluster_run_ids) > max_runs_per_cluster:
            subcluster_run_ids = random.sample(subcluster_run_ids, max_runs_per_cluster)
        
        # Plot sub-cluster trajectories in color
        for run_id in subcluster_run_ids:
            run_df = final_runs_df[final_runs_df["run_id"] == run_id]
            if len(run_df) == 0:
                continue
            coords = run_df[["x_mirror_c", "y_mirror_c"]].values
            if coords.shape[0] >= 2:
                ax.plot(coords[:, 0], coords[:, 1], alpha=0.5, color=colors[i], linewidth=1.5)
        
        # Get tactical label and stats
        tactical_label = tactical_labels.get(subcluster_id, "")
        if len(tactical_label) > 20:
            tactical_label = tactical_label[:20] + "..."
            
        sub_id = subcluster_id.split('_')[1]
        title_text = f"Sub-cluster {original_cluster_id}.{sub_id}\n({len(subcluster_run_ids)} runs)"
        
        ax.set_title(title_text, fontsize=10, fontweight='bold')
        
        if tactical_label:
            ax.text(0.5, -0.1, tactical_label, transform=ax.transAxes,
                   ha="center", va="top", fontsize=8, style='italic',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.subplots_adjust(top=0.90)
    return fig


def create_cluster_tree_visualization(refined_assignments, cluster_analysis, tactical_labels):
    """
    Create a text-based tree visualization of the cluster hierarchy
    """
    
    # Group by base cluster (ensure consistent string conversion)
    hierarchy = {}
    for _, row in refined_assignments.iterrows():
        base_cluster = str(row['base_cluster'])  # Convert to string for consistency
        refined_cluster = str(row['refined_cluster'])
        
        if base_cluster not in hierarchy:
            hierarchy[base_cluster] = []
        if refined_cluster not in hierarchy[base_cluster]:
            hierarchy[base_cluster].append(refined_cluster)
    
    # Sort everything
    for base_cluster in hierarchy:
        hierarchy[base_cluster] = sorted(hierarchy[base_cluster])
    
    print("="*80)
    print("HIERARCHICAL CLUSTER STRUCTURE")
    print("="*80)
    
    for base_cluster in sorted(hierarchy.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
        subclusters = hierarchy[base_cluster]
        n_total_runs = len(refined_assignments[refined_assignments['base_cluster'].astype(str) == base_cluster])
        
        print(f"\n📁 Original Cluster {base_cluster} ({n_total_runs} total runs)")
        
        if len(subclusters) == 1 and subclusters[0].endswith('_0'):
            print("   └── No refinement applied (kept as single cluster)")
        else:
            for i, subcluster in enumerate(subclusters):
                is_last = (i == len(subclusters) - 1)
                connector = "└──" if is_last else "├──"
                
                # Get sub-cluster info
                sub_runs = refined_assignments[refined_assignments['refined_cluster'] == subcluster]
                n_runs = len(sub_runs)
                
                # Get tactical label
                tactical_label = tactical_labels.get(subcluster, "Mixed Tactical Context")
                
                sub_id = subcluster.split('_')[1]
                print(f"   {connector} Sub-cluster {base_cluster}.{sub_id}: {tactical_label} ({n_runs} runs)")
    
    print("="*80)