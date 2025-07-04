import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.interpolate import interp1d

# TACTICAL RUNS: 

def is_overlapping_run(run_df, ball_carrier_df, min_pass_distance=5.0):
    """
    Checks if a run overlaps the ball carrier.
    Skips runs if the ball carrier changes during the run.
    """
    if ball_carrier_df is None:
        return False

    # Check how many unique carriers there were during this run
    unique_carriers = ball_carrier_df["playerId"].nunique()
    # Can look at subRuns later, but for now we assume one carrier per run
    if unique_carriers > 1:
        # Carrier changed mid-run; skip for safety
        return False

    # Proceed as before
    f_start = run_df["frameIdx"].iloc[0]
    f_end = run_df["frameIdx"].iloc[-1]

    runner_start = run_df.iloc[0][["x", "y"]].values
    runner_end = run_df.iloc[-1][["x", "y"]].values

    carrier_start_row = ball_carrier_df.loc[
        ball_carrier_df["frameIdx"] == f_start
    ]

    carrier_end_row = ball_carrier_df.loc[
        ball_carrier_df["frameIdx"] == f_end
    ]

    if carrier_start_row.empty or carrier_end_row.empty:
        return False

    carrier_start = carrier_start_row.iloc[0][["x", "y"]].values
    carrier_end = carrier_end_row.iloc[0][["x", "y"]].values

    # Compute lateral offset (y-direction)
    delta_start_y = runner_start[1] - carrier_start[1]
    delta_end_y = runner_end[1] - carrier_end[1]

    # Check if runner switched sides outside the carrier
    overlap_side_change = np.sign(delta_start_y) != np.sign(delta_end_y)

    # Check lateral distance threshold
    lateral_movement = abs(delta_end_y - delta_start_y)

    # Ensure overlap moves forward enough
    forward_distance = runner_end[0] - runner_start[0]

    return (
        overlap_side_change and
        lateral_movement > 2.0 and
        forward_distance > min_pass_distance
    )

def is_underlapping_run(run_df, ball_carrier_df, min_pass_distance=5.0):
    """
    Heuristic: detects underlapping runs where the runner cuts inside
    relative to the ball carrier.

    Returns True only if the ball carrier remains constant during the run.
    """
    if ball_carrier_df is None:
        return False

    # Check how many unique carriers there were during this run
    unique_carriers = ball_carrier_df["playerId"].nunique()
    if unique_carriers > 1:
        # Carrier changed mid-run; skip for safety
        return False

    # Proceed as before
    f_start = run_df["frameIdx"].iloc[0]
    f_end = run_df["frameIdx"].iloc[-1]

    runner_start = run_df.iloc[0][["x", "y"]].values
    runner_end = run_df.iloc[-1][["x", "y"]].values

    carrier_start_row = ball_carrier_df.loc[
        ball_carrier_df["frameIdx"] == f_start
    ]

    carrier_end_row = ball_carrier_df.loc[
        ball_carrier_df["frameIdx"] == f_end
    ]

    if carrier_start_row.empty or carrier_end_row.empty:
        return False

    carrier_start = carrier_start_row.iloc[0][["x", "y"]].values
    carrier_end = carrier_end_row.iloc[0][["x", "y"]].values

    # Compute lateral offsets (y-difference)
    delta_start_y = runner_start[1] - carrier_start[1]
    delta_end_y = runner_end[1] - carrier_end[1]

    overlap_side_change = np.sign(delta_start_y) != np.sign(delta_end_y)

    lateral_distance_start = abs(delta_start_y)
    lateral_distance_end = abs(delta_end_y)

    lateral_movement = lateral_distance_start - lateral_distance_end

    forward_distance = runner_end[0] - runner_start[0]

    # For underlap:
    # - sign change
    # - lateral distance reduced (runner cuts inside)
    # - forward enough
    return (
        overlap_side_change
        and lateral_movement > 1.0
        and forward_distance > min_pass_distance
    )

def is_diagonal_run(run_df, min_length=5.0, angle_min=20, angle_max=70):
    """
    Heuristic to detect diagonal runs.

    - Checks total length
    - Checks that the angle lies within a diagonal corridor

    Returns True if diagonal.
    """
    # Start and end positions
    runner_start = run_df.iloc[0][["x", "y"]].values
    runner_end = run_df.iloc[-1][["x", "y"]].values

    delta_x = runner_end[0] - runner_start[0]
    delta_y = runner_end[1] - runner_start[1]

    # Total run distance
    total_distance = np.linalg.norm([delta_x, delta_y])
    if total_distance < min_length:
        return False

    # Compute angle in degrees
    angle_deg = np.degrees(np.arctan2(delta_y, delta_x))

    abs_angle = abs(angle_deg)

    # Check if angle is in diagonal corridor
    return angle_min <= abs_angle <= angle_max

def extract_ball_carrier_df_fast(run_df, players_with_ball_df):
    """
    Fast extraction of ball carrier for frames in run_df.
    """
    match_id = run_df["match_id"].iloc[0]
    period = run_df["period"].iloc[0]
    team_role = run_df["team_role"].iloc[0]
    frames = run_df["frameIdx"].unique()

    run_players = players_with_ball_df[
        (players_with_ball_df["match_id"] == match_id) &
        (players_with_ball_df["period"] == period) &
        (players_with_ball_df["frameIdx"].isin(frames))
    ]

    teammates = run_players[run_players["side"] == team_role].copy()

    if teammates.empty:
        return None

    teammates["dist_to_ball"] = np.linalg.norm(
        teammates[["x", "y"]].values - teammates[["ball_x", "ball_y"]].values,
        axis=1
    )

    idx_min_dist = teammates.groupby("frameIdx")["dist_to_ball"].idxmin()
    ball_carrier_df = teammates.loc[idx_min_dist]

    if ball_carrier_df.empty:
        return None
    else:
        return ball_carrier_df
    
# Define grid edges globally
X_EDGES = np.linspace(-52.5, 52.5, 4)
Y_EDGES = np.linspace(-34, 34, 4)

def get_zone(x, y, x_edges=X_EDGES, y_edges=Y_EDGES):
    x_bin = np.digitize([x], x_edges)[0] - 1
    y_bin = np.digitize([y], y_edges)[0] - 1
    x_bin = min(max(x_bin, 0), len(x_edges) - 2)
    y_bin = min(max(y_bin, 0), len(y_edges) - 2)
    zone_idx = y_bin * (len(x_edges) - 1) + x_bin + 1
    return zone_idx

def compute_run_features(final_runs_df, players_with_ball_df):
    """
    Compute all run-level features and tactical labels
    for each run in final_runs_df.

    Returns:
        zones_df: DataFrame with one row per run_id
    """

    zone_records = []

    for run_id, run_df in final_runs_df.groupby("run_id"):

        # Mirrored & centered positions
        start_x_mirror_c = run_df["x_mirror_c"].iloc[0]
        start_y_mirror_c = run_df["y_mirror_c"].iloc[0]

        end_x_mirror_c = run_df["x_mirror_c"].iloc[-1]
        end_y_mirror_c = run_df["y_mirror_c"].iloc[-1]

        coords = run_df[["x", "y"]].values

        if coords.shape[0] < 2:
            run_length = 0.0
        else:
            deltas = np.diff(coords, axis=0)
            segment_lengths = np.linalg.norm(deltas, axis=1)
            run_length = np.sum(segment_lengths)

        mean_speed = run_df["speed"].mean()
        max_speed = run_df["speed"].max()

        dx = end_x_mirror_c - start_x_mirror_c
        dy = end_y_mirror_c - start_y_mirror_c

        run_angle_rad = np.arctan2(dy, dx)
        run_angle_deg = np.degrees(run_angle_rad)

        run_forward = dx > 0  # True if run is forward (right side of pitch)

        start_zone = get_zone(start_x_mirror_c, start_y_mirror_c)
        end_zone = get_zone(end_x_mirror_c, end_y_mirror_c)

        # Absolute pitch positions
        start_x_abs = run_df["x"].iloc[0]
        start_y_abs = run_df["y"].iloc[0]

        end_x_abs = run_df["x"].iloc[-1]
        end_y_abs = run_df["y"].iloc[-1]

        start_zone_abs = get_zone(start_x_abs, start_y_abs)
        end_zone_abs = get_zone(end_x_abs, end_y_abs)

        phase_of_play = run_df["phase_of_play"].iloc[0]
        in_possession = run_df["in_possession"].iloc[0]
        team_role = run_df["team_role"].iloc[0]
        position = run_df["position"].iloc[0]

        # Extract ball carrier df
        ball_carrier_df = extract_ball_carrier_df_fast(
            run_df, players_with_ball_df
        )

        # Check tactical run types
        overlapping = is_overlapping_run(run_df, ball_carrier_df)
        underlapping = is_underlapping_run(run_df, ball_carrier_df)
        is_diag = is_diagonal_run(run_df)

        zone_records.append({
            "run_id": run_id,
            "start_zone": start_zone,
            "end_zone": end_zone,
            "start_zone_absolute": start_zone_abs,
            "end_zone_absolute": end_zone_abs,
            "phase_of_play": phase_of_play,
            "in_possession": in_possession,
            "team_role": team_role,
            "position": position,
            "run_length_m": run_length,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "run_angle_deg": run_angle_deg,
            "run_forward": run_forward,
            "tactical_overlap": overlapping,
            "tactical_underlap": underlapping,
            "tactical_diagonal": is_diag
        })

    zones_df = pd.DataFrame(zone_records)
    return zones_df