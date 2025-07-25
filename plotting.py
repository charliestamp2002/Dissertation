
import matplotlib.pyplot as plt
import random
from bezier_clustering import *

def draw_pitch(ax, pitch_length=105, pitch_width=68):
    """
    Draws a football pitch centered around (0, 0) on the given matplotlib axis.
    """
    half_length = pitch_length / 2
    half_width = pitch_width / 2

    # Outer boundary & centre line
    ax.plot([-half_length, -half_length, half_length, half_length, -half_length],
            [-half_width, half_width, half_width, -half_width, -half_width], color="black")
    ax.plot([0, 0], [-half_width, half_width], color="black")

    # Left penalty area
    ax.plot([-half_length + 16.5, -half_length + 16.5], [-13.84, 13.84], color="black")
    ax.plot([-half_length, -half_length + 16.5], [-13.84, -13.84], color="black")
    ax.plot([-half_length, -half_length + 16.5], [13.84, 13.84], color="black")

    # Right penalty area
    ax.plot([half_length - 16.5, half_length - 16.5], [-13.84, 13.84], color="black")
    ax.plot([half_length, half_length - 16.5], [-13.84, -13.84], color="black")
    ax.plot([half_length, half_length - 16.5], [13.84, 13.84], color="black")

    # Center circle
    circle = plt.Circle((0, 0), 9.15, color="black", fill=False)
    ax.add_patch(circle)

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
    plot_absolute_positions=True,
    start_zones=None,
    end_zones=None,
    phases_of_play=None,
    positions=None,
    use_absolute_zones=False,
    start_zones_absolute=None,
    end_zones_absolute=None,
    run_angle_range = None, #tuple of (min_angle, max_angle) in degrees
    run_forward = None, #bool, whether to filter for forward runs only
    run_length_range = None, #tuple of (min_length, max_length) in meters
    mean_speed_range = None, #tuple of (min_speed, max_speed) in meters per second
    max_speed_range = None, #tuple of (min_speed, max_speed)
    tactical_overlap=None,
    tactical_underlap=None,
    tactical_diagonal=None
):
   
    num_clusters = len(cluster_control_points)
    fig, axes = plt.subplots(7, 10, figsize=(30, 20))
    axes = axes.flatten()

    # --- FILTERING ---
    filtered_assignments = assignments_zones.copy()

    if start_zones is not None and not use_absolute_zones:
        filtered_assignments = filtered_assignments[
            filtered_assignments["start_zone"].isin(start_zones)
        ]

    if end_zones is not None and not use_absolute_zones:
        filtered_assignments = filtered_assignments[
            filtered_assignments["end_zone"].isin(end_zones)
        ]

    if start_zones_absolute is not None and use_absolute_zones:
        filtered_assignments = filtered_assignments[
            filtered_assignments["start_zone_absolute"].isin(start_zones_absolute)
        ]

    if end_zones_absolute is not None and use_absolute_zones:
        filtered_assignments = filtered_assignments[
            filtered_assignments["end_zone_absolute"].isin(end_zones_absolute)
        ]

    if phases_of_play is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["phase_of_play"].isin(phases_of_play)
        ]

    if positions is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["position"].isin(positions)
        ]

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
            (filtered_assignments["run_length_m"] <= max_len)
        ]

    if mean_speed_range is not None:
        min_speed, max_speed = mean_speed_range
        filtered_assignments = filtered_assignments[
            (filtered_assignments["mean_speed"] >= min_speed) &
            (filtered_assignments["mean_speed"] <= max_speed)
        ]

    if max_speed_range is not None:
        min_speed, max_speed = max_speed_range
        filtered_assignments = filtered_assignments[
            (filtered_assignments["max_speed"] >= min_speed) &
            (filtered_assignments["max_speed"] <= max_speed)
        ]

    if tactical_overlap is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["tactical_overlap"] == tactical_overlap
        ]

    if tactical_underlap is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["tactical_underlap"] == tactical_underlap
        ]

    if tactical_diagonal is not None:
        filtered_assignments = filtered_assignments[
            filtered_assignments["tactical_diagonal"] == tactical_diagonal
        ]

    filtered_run_ids = filtered_assignments["run_id"].unique()

    bucket_pivot = bucket_pivot.set_index("assigned_cluster")

    for cluster_idx in range(num_clusters):
        ax = axes[cluster_idx]
        draw_pitch(ax)

        cluster_run_ids = assignments_zones.loc[
            assignments_zones["assigned_cluster"] == cluster_idx, "run_id"
        ].tolist()

        cluster_run_ids = [
            rid for rid in cluster_run_ids if rid in filtered_run_ids
        ]

        if len(cluster_run_ids) > max_runs_per_cluster:
            cluster_run_ids = random.sample(cluster_run_ids, max_runs_per_cluster)

        for run_id in cluster_run_ids:
            run_df = final_runs_df[final_runs_df["run_id"] == run_id]

            coords = run_df[["x_mirror_c", "y_mirror_c"]].values
            if coords.shape[0] < 2:
                continue

            resampled = resample_coords(coords, num_points=50)
            control_pts = fit_bezier_curve(resampled, num_control_points)

            if plot_absolute_positions:
                start_pos = run_df[["x", "y"]].values[0]
                shifted_ctrl_pts = control_pts - control_pts[0] + start_pos
                bezier_curve = evaluate_bezier_curve(
                    shifted_ctrl_pts, num_points=50
                )
            else:
                bezier_curve = evaluate_bezier_curve(
                    control_pts, num_points=50
                )

            ax.plot(bezier_curve[:, 0], bezier_curve[:, 1], alpha=0.2, color="blue")

        if cluster_idx in bucket_pivot.index:
            row = bucket_pivot.loc[cluster_idx]
            text_lines = []
            for bucket in ["attacker", "midfielder", "defender", "sub", "unknown"]:
                if bucket in row and row[bucket] > 0:
                    text_lines.append(f"{bucket}: {int(row[bucket])}")
            text = "\n".join(text_lines)
            ax.text(
                0.5, 1.2, text,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )

        ax.set_title(f"Cluster {cluster_idx}", fontsize=8)

    for i in range(num_clusters, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.suptitle("All Bézier Run Trajectories per Cluster (On-Pitch View)", fontsize=16, y=1.02)
    plt.show()