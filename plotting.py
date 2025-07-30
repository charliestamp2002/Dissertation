
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from matplotlib.patches import FancyBboxPatch, Circle
from bezier_clustering import resample_coords, fit_bezier_curve, evaluate_bezier_curve


def draw_pitch(ax, pitch_length=105, pitch_width=75):
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


def plot_all_cluster_trajectories_on_pitch(
    final_runs_df,
    assignments_zones,
    cluster_control_points,
    bucket_pivot,
    num_control_points=4,
    max_runs_per_cluster=30,
    title=None,
    is_autoencoder=False,
    plot_absolute_positions=True,
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
    num_clusters = len(cluster_control_points)
    fig, axes = plt.subplots(7, 10, figsize=(30, 20))
    axes = axes.flatten()

    filtered_assignments = assignments_zones.copy()

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

    filtered_run_ids = filtered_assignments["run_id"].unique()
    bucket_pivot = bucket_pivot.set_index("assigned_cluster")

    for cluster_idx in range(num_clusters):
        ax = axes[cluster_idx]
        draw_pitch(ax)

        cluster_run_ids = assignments_zones.loc[
            assignments_zones["assigned_cluster"] == cluster_idx, "run_id"].tolist()
        cluster_run_ids = [rid for rid in cluster_run_ids if rid in filtered_run_ids]

        if len(cluster_run_ids) > max_runs_per_cluster:
            cluster_run_ids = random.sample(cluster_run_ids, max_runs_per_cluster)

        for run_id in cluster_run_ids:
            run_df = final_runs_df[final_runs_df["run_id"] == run_id]
            coords = run_df[["x_mirror_c", "y_mirror_c"]].values
            if coords.shape[0] < 2:
                continue

            resampled = resample_coords(coords, num_points=50)
            control_pts = fit_bezier_curve(resampled, num_control_points)

            if is_autoencoder:
                trajectory = resampled
            else:
                if plot_absolute_positions:
                    start_pos = run_df[["x", "y"]].values[0]
                    shifted_ctrl_pts = control_pts - control_pts[0] + start_pos
                    trajectory = evaluate_bezier_curve(shifted_ctrl_pts, num_points=50)
                else:
                    trajectory = evaluate_bezier_curve(control_pts, num_points=50)

            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.2, color="blue")

        if cluster_idx in bucket_pivot.index:
            row = bucket_pivot.loc[cluster_idx]
            text_lines = []
            for bucket in ["attacker", "midfielder", "defender", "sub", "unknown"]:
                if bucket in row and row[bucket] > 0:
                    text_lines.append(f"{bucket}: {int(row[bucket])}")
            text = "\n".join(text_lines)
            ax.text(0.5, 1.2, text, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(f"Cluster {cluster_idx}", fontsize=8)

    for i in range(num_clusters, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.suptitle(f"All {title} Run Trajectories per Cluster (On-Pitch View)", fontsize=16, y=1.02)
    plt.show()


