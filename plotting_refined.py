import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from matplotlib.patches import FancyBboxPatch, Circle
from bezier_clustering import resample_coords, fit_bezier_curve, evaluate_bezier_curve
import pandas as pd


def draw_pitch(ax, pitch_length=105, pitch_width=80):
    half_length = pitch_length / 2
    half_width = pitch_width / 2

    # Professional dark color for pitch lines
    line_color = "#2C3E50"
    
    # Pitch outline and halfway line
    ax.plot([-half_length, -half_length, half_length, half_length, -half_length],
            [-half_width, half_width, half_width, -half_width, -half_width], 
            color=line_color, linewidth=1.5)
    ax.plot([0, 0], [-half_width, half_width], color=line_color, linewidth=1.5)

    # Penalty areas
    ax.plot([-half_length + 16.5, -half_length + 16.5], [-13.84, 13.84], color=line_color, linewidth=1.5)
    ax.plot([-half_length, -half_length + 16.5], [-13.84, -13.84], color=line_color, linewidth=1.5)
    ax.plot([-half_length, -half_length + 16.5], [13.84, 13.84], color=line_color, linewidth=1.5)

    ax.plot([half_length - 16.5, half_length - 16.5], [-13.84, 13.84], color=line_color, linewidth=1.5)
    ax.plot([half_length, half_length - 16.5], [-13.84, -13.84], color=line_color, linewidth=1.5)
    ax.plot([half_length, half_length - 16.5], [13.84, 13.84], color=line_color, linewidth=1.5)

    # Center circle
    circle = plt.Circle((0, 0), 9.15, color=line_color, fill=False, linewidth=1.5)
    ax.add_patch(circle)

    # Add subtle background color for better contrast
    ax.add_patch(plt.Rectangle((-half_length, -half_width), pitch_length, pitch_width,
                              facecolor='#F8F9FA', alpha=0.3, zorder=0))

    ax.set_xlim(-half_length, half_length)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect("equal")
    ax.axis("off")


def get_cluster_colors(n_colors, base_color=None):
    """Generate professional, distinct colors for sub-clusters"""
    if base_color is None:
        # Professional color palette suitable for academic work
        professional_colors = [
            '#2E86AB',  # Professional Blue
            '#A23B72',  # Deep Magenta
            '#F18F01',  # Amber Orange
            '#C73E1D',  # Crimson Red
            '#7209B7',  # Royal Purple
            '#06A77D',  # Teal Green
            '#D864A9',  # Rose Pink
            '#5D737E',  # Slate Gray
            '#91C7B1',  # Sage Green
            '#F4A261',  # Sandy Orange
            '#264653',  # Dark Teal
            '#E76F51',  # Coral
            '#457B9D',  # Steel Blue
            '#A8DADC',  # Light Cyan
            '#1D3557',  # Navy Blue
        ]
        # Cycle through colors if we need more
        return [professional_colors[i % len(professional_colors)] for i in range(n_colors)]
    else:
        # Generate professional shades of a base color
        import matplotlib.colors as mcolors
        base_rgb = mcolors.to_rgb(base_color)
        colors = []
        for i in range(n_colors):
            # Create more subtle variations
            factor = 0.5 + 0.5 * i / max(1, n_colors - 1)  # From medium to dark
            colors.append((base_rgb[0] * factor, base_rgb[1] * factor, base_rgb[2] * factor))
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
    
    # Adjust figure size - increase vertical spacing for original_only mode
    if plot_mode == "original_only":
        base_figsize = (3 * cols, 3.2 * rows)  # Increased height for better spacing
    else:
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
            # Use professional blue for single-color plots
            color = "#2E86AB"

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
                resampled = resample_coords(coords, num_points=25)
                print(f"  Resampled to {len(resampled)} points")
                if len(resampled) >= 4:  # Need at least 4 points for Bézier
                    control_pts = fit_bezier_curve(resampled, num_control_points=4)
                    if plot_absolute_positions:
                        start_pos = run_df[["x", "y"]].values[0]
                        print(f"  Using absolute positions, start_pos: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
                        shifted_ctrl_pts = control_pts - control_pts[0] + start_pos
                        trajectory = evaluate_bezier_curve(shifted_ctrl_pts, num_points=25)
                    else:
                        print(f"  Using relative positions")
                        trajectory = evaluate_bezier_curve(control_pts, num_points=25)
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
                resampled = resample_coords(coords, num_points=25)
                if len(resampled) >= 4:  # Need at least 4 points for Bézier
                    control_pts = fit_bezier_curve(resampled, num_control_points=4)
                    if plot_absolute_positions:
                        start_pos = run_df[["x", "y"]].values[0]
                        shifted_ctrl_pts = control_pts - control_pts[0] + start_pos
                        trajectory = evaluate_bezier_curve(shifted_ctrl_pts, num_points=25)
                    else:
                        trajectory = evaluate_bezier_curve(control_pts, num_points=25)
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
        elif plot_mode == "original_only":
            title_text = f"Cluster {cluster_id}"
        else:
            title_text = str(cluster_id)
            
        # Adjust title size based on plot mode
        if plot_mode == "original_only":
            ax.set_title(title_text, fontsize=14, fontweight='bold')
        else:
            ax.set_title(title_text, fontsize=10, fontweight='bold')
        
        # Add tactical label with professional styling
        if tactical_label:
            ax.text(0.5, 1.15, tactical_label, transform=ax.transAxes, 
                   ha="center", va="bottom", fontsize=8, style='italic',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", 
                            alpha=0.85, edgecolor="#1976D2", linewidth=0.5))
        
        # Add statistics with professional styling - adjust size based on plot mode
        if plot_mode == "original_only":
            ax.text(0.5, -0.15, stats_text, transform=ax.transAxes,
                   ha="center", va="top", fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", 
                            alpha=0.9, edgecolor="#757575", linewidth=0.5))
        else:
            ax.text(0.5, -0.15, stats_text, transform=ax.transAxes,
                   ha="center", va="top", fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", 
                            alpha=0.9, edgecolor="#757575", linewidth=0.5))

    # Hide unused subplots
    for i in range(num_clusters, len(axes)):
        axes[i].axis("off")

    # Adjust spacing based on plot mode
    if plot_mode == "original_only":
        plt.tight_layout(pad=2.0)  # Increased padding for better spacing
        plt.suptitle(plot_title, fontsize=20, y=0.94)
        plt.subplots_adjust(top=0.90, hspace=0.4)  # Increased vertical spacing
    else:
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
            ax.plot(coords[:, 0], coords[:, 1], alpha=0.4, color="#6C757D", linewidth=1)
    
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
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E8", 
                            alpha=0.85, edgecolor="#4CAF50", linewidth=0.5))
    
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


def plot_all_bezier_clusters_overview(
    final_runs_df,
    assignments_zones,
    cluster_control_points,
    max_runs_per_cluster=0,
    show_control_points=True,
    show_center_curves=True,
    title="All Bézier Clusters Overview"
):
    """
    Plot all 70 Bézier clusters on a single pitch showing control points and sample runs
    
    Args:
        final_runs_df: DataFrame with run trajectory data
        assignments_zones: DataFrame with cluster assignments  
        cluster_control_points: Array of control points for each cluster
        max_runs_per_cluster: Maximum number of runs to show per cluster
        show_control_points: Whether to show the control points
        show_center_curves: Whether to show the center Bézier curves
        title: Plot title
    """
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    draw_pitch(ax)
    
    # Get professional colors for clusters
    num_clusters = len(cluster_control_points)
    colors = get_cluster_colors(num_clusters)
    
    print(f"Plotting overview of {num_clusters} Bézier clusters...")
    
    cluster_run_counts = []
    
    for cluster_idx in range(num_clusters):
        color = colors[cluster_idx % len(colors)]
        
        # Get runs for this cluster
        cluster_run_ids = assignments_zones.loc[
            assignments_zones["assigned_cluster"] == cluster_idx, "run_id"].tolist()
        
        if len(cluster_run_ids) == 0:
            cluster_run_counts.append(0)
            continue
            
        cluster_run_counts.append(len(cluster_run_ids))
        
        # Sample runs for display
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

            # Create Bézier trajectory
            resampled = resample_coords(coords, num_points=25)
            if len(resampled) >= 4:
                control_pts = fit_bezier_curve(resampled, num_control_points=4)
                trajectory = evaluate_bezier_curve(control_pts, num_points=25)
            else:
                trajectory = coords

            # Plot trajectory with low alpha for overview
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   alpha=0.15, color=color, linewidth=0.8)
        
        # Plot cluster control points and center curve if available
        if cluster_control_points[cluster_idx] is not None:
            control_pts = cluster_control_points[cluster_idx]
            
            if show_control_points and len(control_pts) >= 2:
                # Plot control points
                ax.scatter(control_pts[:, 0], control_pts[:, 1], 
                         c=[color], s=25, marker='o', alpha=0.8, 
                         edgecolors='white', linewidths=1.0, zorder=5)
            
            if show_center_curves and len(control_pts) >= 2:
                # Plot center Bézier curve
                center_curve = evaluate_bezier_curve(control_pts, num_points=25)
                ax.plot(center_curve[:, 0], center_curve[:, 1], 
                       color=color, linewidth=2.5, alpha=0.9, zorder=4)
    
    # Add title and information
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend information
    total_runs = sum(cluster_run_counts)
    non_empty_clusters = sum(1 for count in cluster_run_counts if count > 0)
    info_text = f"Clusters: {non_empty_clusters}/{num_clusters}\n"
    info_text += f"Total runs: {total_runs:,}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
    verticalalignment='top', fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", 
                    alpha=0.9, edgecolor="#6C757D", linewidth=0.8))
    # else: 
    #     info_text = f"Clusters: {non_empty_clusters}/{num_clusters}\n"
    #     info_text += f"Total runs: {total_runs:,}\n"
    #     info_text += f"Runs shown: ≤{max_runs_per_cluster} per cluster"
    #     ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
    #     verticalalignment='top', fontsize=10,
    #     bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", 
    #                 alpha=0.9, edgecolor="#6C757D", linewidth=0.8))
    
    

    
    # Add legend for visual elements
    legend_elements = []
    if show_center_curves:
        legend_elements.append(plt.Line2D([0], [0], color='#2E86AB', linewidth=2.5, 
                                        label='Cluster center curves'))
    if show_control_points:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='#2E86AB', markersize=8,
                                        markeredgecolor='white', markeredgewidth=1,
                                        label='Control points', linestyle='None'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fontsize=9,
                 fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Print cluster statistics
    print(f"\nCluster Statistics:")
    print(f"  Total clusters: {num_clusters}")
    print(f"  Non-empty clusters: {non_empty_clusters}")
    print(f"  Total runs: {total_runs:,}")
    print(f"  Average runs per cluster: {total_runs/non_empty_clusters:.1f}")
    print(f"  Min runs per cluster: {min([c for c in cluster_run_counts if c > 0])}")
    print(f"  Max runs per cluster: {max(cluster_run_counts)}")
    
    return fig


def plot_single_bezier_cluster(
    final_runs_df,
    assignments_zones,
    cluster_control_points,
    cluster_id,
    max_runs_to_show=15,
    show_control_points=True,
    show_direction_arrows=True,
    title=None
):
    """
    Plot a single Bézier cluster showing the center curve and associated runs
    
    Args:
        final_runs_df: DataFrame with run trajectory data
        assignments_zones: DataFrame with cluster assignments  
        cluster_control_points: Array of control points for each cluster
        cluster_id: ID of the cluster to plot
        max_runs_to_show: Maximum number of runs to display
        show_control_points: Whether to show the 4 control points
        show_direction_arrows: Whether to show direction arrows on trajectories
        title: Custom title (if None, auto-generated)
    """
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_pitch(ax)
    
    # Get runs for this cluster
    cluster_runs = assignments_zones[assignments_zones["assigned_cluster"] == cluster_id]
    cluster_run_ids = cluster_runs["run_id"].tolist()
    
    if len(cluster_run_ids) == 0:
        print(f"Warning: No runs found for cluster {cluster_id}")
        return None
    
    print(f"Plotting cluster {cluster_id} with {len(cluster_run_ids)} runs...")
    
    # Sample runs if too many
    if len(cluster_run_ids) > max_runs_to_show:
        cluster_run_ids = random.sample(cluster_run_ids, max_runs_to_show)
    
    # Plot individual runs in professional blue
    professional_blue = "#2E86AB"
    
    for run_id in cluster_run_ids:
        run_df = final_runs_df[final_runs_df["run_id"] == run_id]
        if len(run_df) == 0:
            continue
            
        coords = run_df[["x_mirror_c", "y_mirror_c"]].values
        if coords.shape[0] < 2:
            continue

        # Create Bézier trajectory
        resampled = resample_coords(coords, num_points=25)
        if len(resampled) >= 4:
            control_pts = fit_bezier_curve(resampled, num_control_points=4)
            trajectory = evaluate_bezier_curve(control_pts, num_points=25)
        else:
            trajectory = coords

        # Plot trajectory in professional blue
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               alpha=0.6, color=professional_blue, linewidth=1.5, zorder=2)
        
        # Add direction arrow if requested
        if show_direction_arrows and len(trajectory) > 1:
            # Add arrow at 70% of the trajectory
            arrow_idx = int(len(trajectory) * 0.7)
            if arrow_idx < len(trajectory) - 1:
                start_point = trajectory[arrow_idx]
                end_point = trajectory[arrow_idx + 1]
                
                # Calculate arrow direction
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                
                ax.annotate('', xy=end_point, xytext=start_point,
                           arrowprops=dict(arrowstyle='->', color=professional_blue, 
                                         lw=1.5, alpha=0.8), zorder=3)
    
    # Plot cluster center curve in black
    if cluster_control_points[cluster_id] is not None:
        control_pts = cluster_control_points[cluster_id]
        
        if len(control_pts) >= 2:
            # Plot center Bézier curve in black
            center_curve = evaluate_bezier_curve(control_pts, num_points=25)
            ax.plot(center_curve[:, 0], center_curve[:, 1], 
                   color='black', linewidth=3.5, alpha=0.9, zorder=4,
                   label='Cluster center curve')
            
            # Add direction arrow for center curve
            if show_direction_arrows and len(center_curve) > 1:
                arrow_idx = int(len(center_curve) * 0.7)
                if arrow_idx < len(center_curve) - 1:
                    start_point = center_curve[arrow_idx]
                    end_point = center_curve[arrow_idx + 1]
                    
                    ax.annotate('', xy=end_point, xytext=start_point,
                               arrowprops=dict(arrowstyle='->', color='black', 
                                             lw=2.5, alpha=0.9), zorder=5)
            
            # Plot control points if requested
            if show_control_points:
                ax.scatter(control_pts[:, 0], control_pts[:, 1], 
                         c='#6C757D', s=50, marker='o', alpha=0.9, 
                         edgecolors='white', linewidths=1.5, zorder=6,
                         label='Control points')
                
                # Number the control points
                for i, (x, y) in enumerate(control_pts):
                    ax.annotate(f'{i+1}', (x, y), xytext=(3, 3), 
                               textcoords='offset points', fontsize=9, 
                               fontweight='bold', color='white', zorder=7)
    
    # Set title
    if title is None:
        title = f"Bézier Cluster {cluster_id} - Center Curve and Associated Runs"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add cluster information
    info_text = f"Cluster {cluster_id}\n"
    info_text += f"Total runs: {len(cluster_runs)}\n"
    info_text += f"Runs shown: {len(cluster_run_ids)}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", 
                    alpha=0.9, edgecolor="#6C757D", linewidth=0.8))
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=3.5, 
                   label='Cluster center curve'),
        plt.Line2D([0], [0], color=professional_blue, linewidth=1.5, alpha=0.6,
                   label='Associated runs')
    ]
    
    if show_control_points:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#6C757D', markersize=8,
                      markeredgecolor='white', markeredgewidth=1.5,
                      label='Control points', linestyle='None')
        )
    
    if show_direction_arrows:
        legend_elements.append(
            plt.Line2D([0], [0], color='gray', 
                      marker='>', markersize=8, linestyle='None',
                      label='Direction arrows')
        )
    
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(0.98, 0.85), fontsize=10,
             fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Print cluster statistics
    print(f"\nCluster {cluster_id} Details:")
    print(f"  Total runs in cluster: {len(cluster_runs)}")
    print(f"  Runs displayed: {len(cluster_run_ids)}")
    if cluster_control_points[cluster_id] is not None:
        control_pts = cluster_control_points[cluster_id]
        print(f"  Control points: {len(control_pts)}")
        print(f"  Center curve start: ({control_pts[0, 0]:.1f}, {control_pts[0, 1]:.1f})")
        print(f"  Center curve end: ({control_pts[-1, 0]:.1f}, {control_pts[-1, 1]:.1f})")
    
    return fig


def compute_cluster_center_trajectory(final_runs_df, assignments_zones, cluster_id, method="median", num_points=50):
    """
    Compute a representative center trajectory for autoencoder/transformer clusters
    
    Args:
        final_runs_df: DataFrame with run trajectory data
        assignments_zones: DataFrame with cluster assignments
        cluster_id: ID of the cluster
        method: "median", "mean", or "medoid" - how to compute center
        num_points: Number of points for the output trajectory
    
    Returns:
        center_trajectory: Array of (x, y) points representing cluster center
    """
    
    # Get cluster column name based on method
    if "ae_cluster" in assignments_zones.columns:
        cluster_col = "ae_cluster"
    elif "transformer_cluster" in assignments_zones.columns:
        cluster_col = "transformer_cluster"
    elif "assigned_cluster" in assignments_zones.columns:
        cluster_col = "assigned_cluster"
    else:
        raise ValueError("Could not find cluster assignment column")
    
    # Get runs for this cluster
    cluster_runs = assignments_zones[assignments_zones[cluster_col] == cluster_id]
    cluster_run_ids = cluster_runs["run_id"].tolist()
    
    if len(cluster_run_ids) == 0:
        return None
    
    # Collect all trajectories and resample to same length
    trajectories = []
    for run_id in cluster_run_ids:
        run_df = final_runs_df[final_runs_df["run_id"] == run_id]
        if len(run_df) == 0:
            continue
            
        coords = run_df[["x_mirror_c", "y_mirror_c"]].values
        if coords.shape[0] < 2:
            continue
        
        # Resample to consistent length
        resampled = resample_coords(coords, num_points=num_points)
        if len(resampled) == num_points:
            trajectories.append(resampled)
    
    if len(trajectories) == 0:
        return None
    
    trajectories = np.array(trajectories)  # Shape: (n_runs, num_points, 2)
    
    if method == "mean":
        # Simple mean across all trajectories
        center_trajectory = np.mean(trajectories, axis=0)
    elif method == "median":
        # Element-wise median across all trajectories  
        center_trajectory = np.median(trajectories, axis=0)
    elif method == "medoid":
        # Find the trajectory closest to the mean (medoid)
        mean_traj = np.mean(trajectories, axis=0)
        distances = []
        for traj in trajectories:
            dist = np.sum(np.sqrt(np.sum((traj - mean_traj)**2, axis=1)))
            distances.append(dist)
        medoid_idx = np.argmin(distances)
        center_trajectory = trajectories[medoid_idx]
    else:
        raise ValueError("Method must be 'mean', 'median', or 'medoid'")
    
    return center_trajectory


def plot_single_autoencoder_transformer_cluster(
    final_runs_df,
    assignments_zones,
    cluster_id,
    method_name="Autoencoder",  # "Autoencoder" or "Transformer"
    center_method="median",  # "mean", "median", or "medoid"
    max_runs_to_show=15,
    show_center_points=True,
    show_direction_arrows=True,
    title=None
):
    """
    Plot a single Autoencoder/Transformer cluster with computed center trajectory
    
    Args:
        final_runs_df: DataFrame with run trajectory data
        assignments_zones: DataFrame with cluster assignments
        cluster_id: ID of the cluster to plot
        method_name: "Autoencoder" or "Transformer" for labeling
        center_method: How to compute center ("mean", "median", "medoid")
        max_runs_to_show: Maximum number of runs to display
        show_center_points: Whether to show points along the center trajectory
        show_direction_arrows: Whether to show direction arrows
        title: Custom title (if None, auto-generated)
    """
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_pitch(ax)
    
    # Get cluster column name
    if "ae_cluster" in assignments_zones.columns:
        cluster_col = "ae_cluster"
    elif "transformer_cluster" in assignments_zones.columns:
        cluster_col = "transformer_cluster"
    elif "assigned_cluster" in assignments_zones.columns:
        cluster_col = "assigned_cluster"
    else:
        print("Error: Could not find cluster assignment column")
        return None
    
    # Get runs for this cluster
    cluster_runs = assignments_zones[assignments_zones[cluster_col] == cluster_id]
    cluster_run_ids = cluster_runs["run_id"].tolist()
    
    if len(cluster_run_ids) == 0:
        print(f"Warning: No runs found for cluster {cluster_id}")
        return None
    
    print(f"Plotting {method_name} cluster {cluster_id} with {len(cluster_run_ids)} runs...")
    
    # Sample runs if too many
    if len(cluster_run_ids) > max_runs_to_show:
        cluster_run_ids = random.sample(cluster_run_ids, max_runs_to_show)
    
    # Plot individual runs in professional blue
    professional_blue = "#2E86AB"
    
    for run_id in cluster_run_ids:
        run_df = final_runs_df[final_runs_df["run_id"] == run_id]
        if len(run_df) == 0:
            continue
            
        coords = run_df[["x_mirror_c", "y_mirror_c"]].values
        if coords.shape[0] < 2:
            continue

        # Plot trajectory in professional blue (no Bézier fitting for AE/Transformer)
        ax.plot(coords[:, 0], coords[:, 1], 
               alpha=0.6, color=professional_blue, linewidth=1.5, zorder=2)
        
        # Add direction arrow if requested
        if show_direction_arrows and len(coords) > 1:
            # Add arrow at 70% of the trajectory
            arrow_idx = int(len(coords) * 0.7)
            if arrow_idx < len(coords) - 1:
                start_point = coords[arrow_idx]
                end_point = coords[arrow_idx + 1]
                
                ax.annotate('', xy=end_point, xytext=start_point,
                           arrowprops=dict(arrowstyle='->', color=professional_blue, 
                                         lw=1.5, alpha=0.8), zorder=3)
    
    # Compute and plot cluster center trajectory
    center_trajectory = compute_cluster_center_trajectory(
        final_runs_df, assignments_zones, cluster_id, method=center_method
    )
    
    if center_trajectory is not None:
        # Plot center trajectory in black
        ax.plot(center_trajectory[:, 0], center_trajectory[:, 1], 
               color='black', linewidth=3.5, alpha=0.9, zorder=4,
               label=f'Cluster center ({center_method})')
        
        # Add direction arrow for center trajectory
        if show_direction_arrows and len(center_trajectory) > 1:
            arrow_idx = int(len(center_trajectory) * 0.7)
            if arrow_idx < len(center_trajectory) - 1:
                start_point = center_trajectory[arrow_idx]
                end_point = center_trajectory[arrow_idx + 1]
                
                ax.annotate('', xy=end_point, xytext=start_point,
                           arrowprops=dict(arrowstyle='->', color='black', 
                                         lw=2.5, alpha=0.9), zorder=5)
        
        # Plot center points if requested (every 5th point)
        if show_center_points:
            step = max(1, len(center_trajectory) // 10)  # Show ~10 points
            center_points = center_trajectory[::step]
            ax.scatter(center_points[:, 0], center_points[:, 1], 
                     c='#6C757D', s=50, marker='o', alpha=0.9, 
                     edgecolors='white', linewidths=1.5, zorder=6,
                     label='Center trajectory points')
    
    # Set title
    if title is None:
        title = f"{method_name} Cluster {cluster_id} - Center Trajectory and Associated Runs"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add cluster information
    info_text = f"Cluster {cluster_id}\n"
    info_text += f"Total runs: {len(cluster_runs)}\n"
    info_text += f"Runs shown: {len(cluster_run_ids)}\n"
    info_text += f"Center method: {center_method}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", 
                    alpha=0.9, edgecolor="#6C757D", linewidth=0.8))
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=3.5, 
                   label=f'Cluster center ({center_method})'),
        plt.Line2D([0], [0], color=professional_blue, linewidth=1.5, alpha=0.6,
                   label='Associated runs')
    ]
    
    if show_center_points and center_trajectory is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#6C757D', markersize=8,
                      markeredgecolor='white', markeredgewidth=1.5,
                      label='Center trajectory points', linestyle='None')
        )
    
    if show_direction_arrows:
        legend_elements.append(
            plt.Line2D([0], [0], color='gray', 
                      marker='>', markersize=8, linestyle='None',
                      label='Direction arrows')
        )
    
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(0.98, 0.85), fontsize=10,
             fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Print cluster statistics
    print(f"\n{method_name} Cluster {cluster_id} Details:")
    print(f"  Total runs in cluster: {len(cluster_runs)}")
    print(f"  Runs displayed: {len(cluster_run_ids)}")
    print(f"  Center computation method: {center_method}")
    
    if center_trajectory is not None:
        print(f"  Center trajectory points: {len(center_trajectory)}")
        print(f"  Start position: ({center_trajectory[0, 0]:.1f}, {center_trajectory[0, 1]:.1f})")
        print(f"  End position: ({center_trajectory[-1, 0]:.1f}, {center_trajectory[-1, 1]:.1f})")
        
        # Calculate trajectory length
        trajectory_length = 0
        for i in range(len(center_trajectory) - 1):
            dx = center_trajectory[i+1, 0] - center_trajectory[i, 0]
            dy = center_trajectory[i+1, 1] - center_trajectory[i, 1]
            trajectory_length += np.sqrt(dx**2 + dy**2)
        
        print(f"  Center trajectory length: {trajectory_length:.1f}m")
    
    return fig


def plot_all_ae_transformer_clusters_overview(
    final_runs_df,
    assignments_zones,
    method_name="Autoencoder",  # "Autoencoder" or "Transformer"
    center_method="median",  # "mean", "median", or "medoid"
    max_runs_per_cluster=2,
    show_center_trajectories=True,
    show_center_points=True,
    title=None
):
    """
    Plot all Autoencoder/Transformer cluster center trajectories on a single pitch
    
    Args:
        final_runs_df: DataFrame with run trajectory data
        assignments_zones: DataFrame with cluster assignments
        method_name: "Autoencoder" or "Transformer" for labeling
        center_method: How to compute centers ("mean", "median", "medoid")
        max_runs_per_cluster: Maximum sample runs to show per cluster
        show_center_trajectories: Whether to show center trajectories
        show_center_points: Whether to show points along center trajectories
        title: Custom title (if None, auto-generated)
    """
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    draw_pitch(ax)
    
    # Get cluster column name
    if "ae_cluster" in assignments_zones.columns:
        cluster_col = "ae_cluster"
    elif "transformer_cluster" in assignments_zones.columns:
        cluster_col = "transformer_cluster"
    elif "assigned_cluster" in assignments_zones.columns:
        cluster_col = "assigned_cluster"
    else:
        print("Error: Could not find cluster assignment column")
        return None
    
    # Get all unique clusters
    unique_clusters = sorted(assignments_zones[cluster_col].unique())
    num_clusters = len(unique_clusters)
    
    # Get professional colors for clusters
    colors = get_cluster_colors(num_clusters)
    
    print(f"Plotting overview of {num_clusters} {method_name} cluster centers...")
    
    cluster_run_counts = []
    successful_centers = 0
    
    for i, cluster_id in enumerate(unique_clusters):
        color = colors[i % len(colors)]
        
        # Get runs for this cluster
        cluster_runs = assignments_zones[assignments_zones[cluster_col] == cluster_id]
        cluster_run_ids = cluster_runs["run_id"].tolist()
        
        if len(cluster_run_ids) == 0:
            cluster_run_counts.append(0)
            continue
            
        cluster_run_counts.append(len(cluster_run_ids))
        
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
            final_runs_df, assignments_zones, cluster_id, method=center_method
        )
        
        if center_trajectory is not None:
            successful_centers += 1
            
            if show_center_trajectories:
                # Plot center trajectory
                ax.plot(center_trajectory[:, 0], center_trajectory[:, 1], 
                       color=color, linewidth=2.0, alpha=0.8, zorder=3)
            
            if show_center_points:
                # Plot center points (every 5th point)
                step = max(1, len(center_trajectory) // 8)  # Show ~8 points per trajectory
                center_points = center_trajectory[::step]
                ax.scatter(center_points[:, 0], center_points[:, 1], 
                         c=[color], s=15, marker='o', alpha=0.7, 
                         edgecolors='white', linewidths=0.5, zorder=4)
    
    # Set title
    if title is None:
        title = f"All {method_name} Cluster Centers Overview ({center_method.capitalize()} Method)"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add information box
    total_runs = sum(cluster_run_counts)
    non_empty_clusters = sum(1 for count in cluster_run_counts if count > 0)
    
    info_text = f"Clusters: {non_empty_clusters}/{num_clusters}\n"
    info_text += f"Centers computed: {successful_centers}\n"
    info_text += f"Total runs: {total_runs:,}\n"
    info_text += f"Center method: {center_method}"
    if max_runs_per_cluster > 0:
        info_text += f"\nRuns shown: ≤{max_runs_per_cluster} per cluster"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", 
                    alpha=0.9, edgecolor="#6C757D", linewidth=0.8))
    
    # Add legend
    legend_elements = []
    if show_center_trajectories:
        legend_elements.append(
            plt.Line2D([0], [0], color='#2E86AB', linewidth=2.0, alpha=0.8,
                      label=f'Cluster center trajectories ({center_method})')
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
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fontsize=9,
                 fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n{method_name} Cluster Centers Overview Statistics:")
    print(f"  Total clusters: {num_clusters}")
    print(f"  Non-empty clusters: {non_empty_clusters}")
    print(f"  Successful center computations: {successful_centers}")
    print(f"  Total runs: {total_runs:,}")
    print(f"  Center computation method: {center_method}")
    if non_empty_clusters > 0:
        print(f"  Average runs per cluster: {total_runs/non_empty_clusters:.1f}")
        print(f"  Min runs per cluster: {min([c for c in cluster_run_counts if c > 0])}")
        print(f"  Max runs per cluster: {max(cluster_run_counts)}")
    
    return fig