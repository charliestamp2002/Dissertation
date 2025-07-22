import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from data_utils import (extract_surname,
                         find_events_during_run)  # Required to resolve surname lookups from metadata_dict

def resample_coords(coords, num_points=50):
    if len(coords) < 2:
        return np.tile(coords[0], (num_points, 1))
    distances = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    distances = np.insert(distances, 0, 0.0)
    total_length = distances[-1]
    if total_length == 0:
        return np.tile(coords[0], (num_points, 1))
    normalized_dist = distances / total_length
    interp_func = interp1d(normalized_dist, coords, axis=0, kind='linear')
    uniform_dist = np.linspace(0, 1, num_points)
    return interp_func(uniform_dist)

def segment_runs(players_df, speed_threshold=2.0):
    runs = []
    for (match_id, period, playerId), group in players_df.groupby(["match_id", "period", "playerId"]):
        group = group.sort_values("frameIdx")
        current_run = []
        for _, row in group.iterrows():
            if row["speed"] > speed_threshold:
                current_run.append(row)
            elif current_run:
                runs.append(pd.DataFrame(current_run))
                current_run = []
        if current_run:
            runs.append(pd.DataFrame(current_run))
    return runs

def filter_off_ball_runs_with_distance(runs_list, frames_df, min_distance=3.0):
    frame_last_touch = frames_df.set_index(["match_id", "period", "frameIdx"])["lastTouch_team"].to_dict()
    ball_positions = frames_df.set_index(["match_id", "period", "frameIdx"])[["ball_x", "ball_y"]].to_dict("index")

    off_ball_runs = []
    for run_df in runs_list:
        player_id = run_df["playerId"].iloc[0]
        match_id = run_df["match_id"].iloc[0]
        period = run_df["period"].iloc[0]
        frame_idxs = run_df["frameIdx"].values

        is_off_ball = True
        for frame_idx in frame_idxs:
            key = (match_id, period, frame_idx)
            if frame_last_touch.get(key) == player_id:
                is_off_ball = False
                break
            ball_pos = ball_positions.get(key)
            if ball_pos is None:
                continue
            player_pos = run_df[run_df["frameIdx"] == frame_idx][["x", "y"]].values
            if player_pos.size == 0:
                continue
            dist = np.linalg.norm(player_pos[0] - np.array([ball_pos["ball_x"], ball_pos["ball_y"]]))
            if dist < min_distance:
                is_off_ball = False
                break
        if is_off_ball:
            off_ball_runs.append(run_df)
    return off_ball_runs

def mirror_group(group):
    y_mean = group["y"].mean()
    group["y_mirror"] = -group["y"] if y_mean < 0 else group["y"]
    group["x_mirror"] = group["x"]
    return group

def should_flip_x(team_role, period):
    if period == 1:
        return team_role == "away"
    elif period == 2:
        return team_role == "home"
    return False

def simplify_name(name):
    if name is None:
        return None
    parts = name.lower().strip().split()
    return f"{parts[0]} {parts[-1]}" if len(parts) >= 2 else name.lower().strip()

def compute_centroids(players_df):
    players_df["number"] = players_df["number"].astype(int)
    centroids = {}
    for (match_id, period, frame_idx), group in players_df.groupby(["match_id", "period", "frameIdx"]):
        for side in ["home", "away"]:
            team_players = group[(group["side"] == side) & (group["number"] != 1)]
            centroid = team_players[["x", "y"]].mean().values if not team_players.empty else np.array([0.0, 0.0])
            centroids[(match_id, period, frame_idx, side)] = centroid
    return centroids

def process_runs(players_df, frames_df, runs_list):
    # Add metadata fields to each run and assign run_id
    annotated_runs = []
    for i, run_df in enumerate(runs_list):
        run_df = run_df.copy()
        meta_fields = ["playerId", "optaId", "match_id", "player_name", "position", "team_role"]
        for field in meta_fields:
            run_df[field] = run_df.iloc[0][field]
        run_df["run_id"] = i
        annotated_runs.append(run_df)
    all_runs_df = pd.concat(annotated_runs, ignore_index=True)
    all_runs_df = all_runs_df.groupby("run_id", group_keys=False).apply(mirror_group)

    # Prepare lookups
    centroid_dict = compute_centroids(players_df)
    frame_last_touch_team = frames_df.set_index(["match_id", "period", "frameIdx"])["lastTouch_team"].to_dict()

    adjusted_runs = []
    grouped = all_runs_df.groupby(["match_id", "period", "playerId", "run_id"], group_keys=False)

    for _, run_df in grouped:
        run_df = run_df.sort_values("frameIdx")
        match_id = run_df["match_id"].iloc[0]
        period = run_df["period"].iloc[0]
        start_frame = run_df["frameIdx"].iloc[0]
        team_role = run_df["team_role"].iloc[0]
        possession_side = frame_last_touch_team.get((match_id, period, start_frame))

        if possession_side is None:
            in_possession, phase_of_play = np.nan, np.nan
        else:
            in_possession = (team_role == possession_side)
            phase_of_play = "attack" if in_possession else "defend"

        team_centroid = centroid_dict.get((match_id, period, start_frame, team_role), np.array([0.0, 0.0]))

        run_df["in_possession"] = in_possession
        run_df["phase_of_play"] = phase_of_play
        run_df["x_c"] = run_df["x"] - team_centroid[0]
        run_df["y_c"] = run_df["y"] - team_centroid[1]
        run_df["x_mirror_c"] = run_df["x_mirror"] - team_centroid[0]
        run_df["y_mirror_c"] = run_df["y_mirror"] - team_centroid[1]

        if should_flip_x(team_role, period):
            run_df["x_mirror"] = -run_df["x_mirror"]
            run_df["x_mirror_c"] = -run_df["x_mirror_c"]

        adjusted_runs.append(run_df)

    final_runs_df = pd.concat(adjusted_runs, ignore_index=True)
    return adjusted_runs, final_runs_df


# --------- BALL CARRIER + RUN TYPE DETECTION UTILS ---------

def extract_ball_carrier_df_fast(run_df, players_with_ball_df):
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
    return teammates.loc[idx_min_dist] if not teammates.loc[idx_min_dist].empty else None

def is_overlapping_run(run_df, ball_carrier_df, min_pass_distance=5.0, lateral_number=0.5, deg=45):
    if ball_carrier_df is None:
        return False

    unique_carriers = ball_carrier_df["playerId"].nunique()
    if unique_carriers > 1:
        carrier_id = ball_carrier_df["playerId"].mode()[0]
        ball_carrier_df = ball_carrier_df[ball_carrier_df["playerId"] == carrier_id]

    f_start, f_end = run_df["frameIdx"].iloc[0], run_df["frameIdx"].iloc[-1]
    runner_start = run_df.iloc[0][["x_mirror_c", "y_mirror_c"]].values
    runner_end = run_df.iloc[-1][["x_mirror_c", "y_mirror_c"]].values

    carrier_start_row = ball_carrier_df[ball_carrier_df["frameIdx"] == f_start]
    carrier_end_row = ball_carrier_df[ball_carrier_df["frameIdx"] == f_end]

    if carrier_start_row.empty or carrier_end_row.empty:
        return False

    carrier_start = carrier_start_row.iloc[0][["x_mirror_c", "y_mirror_c"]].values
    carrier_end = carrier_end_row.iloc[0][["x_mirror_c", "y_mirror_c"]].values

    delta_start_y = runner_start[1] - carrier_start[1]
    delta_end_y = runner_end[1] - carrier_end[1]

    overlap_side_change = np.sign(delta_start_y) != np.sign(delta_end_y)
    lateral_movement = abs(delta_end_y - delta_start_y)
    forward_distance = runner_end[0] - runner_start[0]

    runner_vec = runner_end - runner_start
    carrier_vec = carrier_end - carrier_start

    if np.linalg.norm(runner_vec) < 1e-6 or np.linalg.norm(carrier_vec) < 1e-6:
        return False

    runner_vec /= np.linalg.norm(runner_vec) + 1e-6
    carrier_vec /= np.linalg.norm(carrier_vec) + 1e-6

    angle_deg = np.degrees(np.arccos(np.clip(np.dot(runner_vec, carrier_vec), -1, 1)))

    return (
        overlap_side_change and
        lateral_movement > lateral_number and
        forward_distance > min_pass_distance and
        angle_deg < deg
    )

def is_underlapping_run(run_df, ball_carrier_df, min_pass_distance=5.0, lateral_number=0.5, deg=120.0):
    if ball_carrier_df is None:
        return False

    unique_carriers = ball_carrier_df["playerId"].nunique()
    if unique_carriers > 1:
        carrier_id = ball_carrier_df["playerId"].mode()[0]
        ball_carrier_df = ball_carrier_df[ball_carrier_df["playerId"] == carrier_id]

    f_start, f_end = run_df["frameIdx"].iloc[0], run_df["frameIdx"].iloc[-1]
    runner_start = run_df.iloc[0][["x_mirror_c", "y_mirror_c"]].values
    runner_end = run_df.iloc[-1][["x_mirror_c", "y_mirror_c"]].values

    carrier_start_row = ball_carrier_df[ball_carrier_df["frameIdx"] == f_start]
    carrier_end_row = ball_carrier_df[ball_carrier_df["frameIdx"] == f_end]

    if carrier_start_row.empty or carrier_end_row.empty:
        return False

    carrier_start = carrier_start_row.iloc[0][["x_mirror_c", "y_mirror_c"]].values
    carrier_end = carrier_end_row.iloc[0][["x_mirror_c", "y_mirror_c"]].values

    delta_start_y = runner_start[1] - carrier_start[1]
    delta_end_y = runner_end[1] - carrier_end[1]

    overlap_side_change = np.sign(delta_start_y) != np.sign(delta_end_y)
    lateral_movement = abs(delta_start_y) - abs(delta_end_y)
    forward_distance = runner_end[0] - runner_start[0]

    runner_vec = runner_end - runner_start
    carrier_vec = carrier_end - carrier_start

    if np.linalg.norm(runner_vec) < 1e-6 or np.linalg.norm(carrier_vec) < 1e-6:
        return False

    runner_vec /= np.linalg.norm(runner_vec) + 1e-6
    carrier_vec /= np.linalg.norm(carrier_vec) + 1e-6

    angle_deg = np.degrees(np.arccos(np.clip(np.dot(runner_vec, carrier_vec), -1, 1)))

    return (
        overlap_side_change and
        lateral_movement > lateral_number and
        forward_distance > min_pass_distance and
        angle_deg < deg
    )

def is_diagonal_run(run_df, min_length=5.0, angle_min=20, angle_max=70):
    runner_start = run_df.iloc[0][["x", "y"]].values
    runner_end = run_df.iloc[-1][["x", "y"]].values

    delta_x = runner_end[0] - runner_start[0]
    delta_y = runner_end[1] - runner_start[1]

    total_distance = np.linalg.norm([delta_x, delta_y])
    if total_distance < min_length:
        return False

    angle_deg = np.degrees(np.arctan2(delta_y, delta_x))
    return angle_min <= abs(angle_deg) <= angle_max

from data_utils import extract_surname  # Required to resolve surname lookups from metadata_dict

def runner_receives_pass(
    run_df,
    ball_carrier_df,
    events_df,
    players_df,
    match_id,
    metadata_dict,
    debug=False,
    debug_further=False
):
    """
    Checks whether the player making the run received
    a pass from the ball carrier during this run.

    This version focuses on surname-to-metadata logic and early exit tracing.
    """

    if ball_carrier_df is None:
        if debug_further:
            print("[EXIT] ball_carrier_df is None")
        return False

    players_df_match = players_df[players_df["match_id"] == match_id]
    tracking_to_opta = dict(
        zip(players_df_match["playerId"].astype(str), players_df_match["optaId"].astype(str))
    )

    unique_carriers = ball_carrier_df["playerId"].unique()

    if len(unique_carriers) == 0:
        if debug_further:
            print(f"[EXIT] No carriers found")
        return False

    carrier_tracking_id = ball_carrier_df["playerId"].value_counts().idxmax()
    carrier_opta_id = tracking_to_opta.get(str(carrier_tracking_id), None)

    if carrier_opta_id is None:
        if debug_further:
            print(f"[EXIT] Could not find Opta ID for tracking ID: {carrier_tracking_id}")
        return False

    runner_optaId = str(run_df["optaId"].iloc[0])
    overlapping_events = find_events_during_run(run_df, events_df)

    for _, e in overlapping_events.iterrows():
        event_type = e.get("type")
        if not isinstance(event_type, dict) or event_type.get("name") != "Pass":
            continue

        event_player = e.get("player", {})
        if not isinstance(event_player, dict):
            if debug_further:
                print("[SKIP] Event player is not a dict")
            continue

        pass_data = e.get("pass", None)
        if pass_data is None:
            if debug_further:
                print("[SKIP] No 'pass' dictionary in event")
            continue

        recipient_info = pass_data.get("recipient")
        if recipient_info is None:
            if debug_further:
                print("[SKIP] No recipient info in 'pass' dict")
            continue

        recipient_name_raw = recipient_info.get("name")
        if recipient_name_raw is None:
            if debug_further:
                print("[SKIP] Recipient name is None")
            continue

        recipient_surname = extract_surname(recipient_name_raw)

        if debug_further:
            if recipient_surname not in metadata_dict:
                print(f"[WARN] Surname '{recipient_surname}' NOT found in metadata_dict")
                print(f"[DEBUG] Sample keys: {list(metadata_dict.keys())[:10]}")
            else:
                print(f"[DEBUG] Surname '{recipient_surname}' found in metadata_dict")

        recipient_optaId = metadata_dict.get(recipient_surname)

        if debug_further:
            print(f"  Recipient name raw: {recipient_name_raw}")
            print(f"  Extracted surname: {recipient_surname}")
            print(f"  Lookup optaId: {recipient_optaId}")
            print(f"  Runner optaId: {runner_optaId}")
            print(f"  Match?: {recipient_optaId == runner_optaId}")

        if recipient_optaId is None:
            if debug_further:
                print("[SKIP] Recipient Opta ID not found in metadata")
            continue

        if recipient_optaId == runner_optaId:
            if debug_further:
                print("[MATCH] Runner received pass from ball carrier")
            return True

    return False


def get_zone(x, y, x_edges, y_edges):
    x_bin = np.digitize([x], x_edges)[0] - 1
    y_bin = np.digitize([y], y_edges)[0] - 1
    x_bin = min(max(x_bin, 0), len(x_edges) - 2)
    y_bin = min(max(y_bin, 0), len(y_edges) - 2)
    zone_idx = y_bin * (len(x_edges) - 1) + x_bin + 1
    return zone_idx


def extract_zone_and_run_features(final_runs_df, players_df, frames_df, events_dfs_by_match, metadata_dict):
    x_edges = np.linspace(-52.5, 52.5, 4)
    y_edges = np.linspace(-34, 34, 4)

    from run_segmentation import (
        extract_ball_carrier_df_fast,
        is_overlapping_run,
        is_underlapping_run,
        is_diagonal_run,
        runner_receives_pass,
        mirror_group,
        should_flip_x,
        compute_centroids
    )

    centroid_dict = compute_centroids(players_df)

    players_with_ball_df = players_df.merge(
        frames_df[["match_id", "period", "frameIdx", "ball_x", "ball_y", "lastTouch_team"]],
        on=["match_id", "period", "frameIdx"],
        how="left",
        suffixes=("", "_ball")
    )

    players_with_ball_df = players_with_ball_df.groupby(
        ["match_id", "period", "playerId"], group_keys=False
    ).apply(mirror_group)

    adjusted_ball_players_list = []

    grouped = players_with_ball_df.groupby(
        ["match_id", "period", "playerId"], group_keys=False
    )

    for _, player_df in grouped:
        player_df = player_df.sort_values("frameIdx")
        match_id = player_df["match_id"].iloc[0]
        period = player_df["period"].iloc[0]
        start_frame = player_df["frameIdx"].iloc[0]
        team_role = player_df["side"].iloc[0]

        team_centroid = centroid_dict.get(
            (match_id, period, start_frame, team_role),
            np.array([0.0, 0.0])
        )

        player_df["x_c"] = player_df["x"] - team_centroid[0]
        player_df["y_c"] = player_df["y"] - team_centroid[1]
        player_df["x_mirror_c"] = player_df["x_mirror"] - team_centroid[0]
        player_df["y_mirror_c"] = player_df["y_mirror"] - team_centroid[1]

        flip_x = should_flip_x(team_role, period)
        if flip_x:
            player_df["x_mirror"] = -player_df["x_mirror"]
            player_df["x_mirror_c"] = -player_df["x_mirror_c"]

        adjusted_ball_players_list.append(player_df)

    players_with_ball_df = pd.concat(adjusted_ball_players_list, ignore_index=True)

    zone_records = []

    for run_id, run_df in final_runs_df.groupby("run_id"):
        match_id = run_df["match_id"].iloc[0]
        events_df = events_dfs_by_match[match_id]

        start_x_mirror_c = run_df["x_mirror_c"].iloc[0]
        start_y_mirror_c = run_df["y_mirror_c"].iloc[0]
        end_x_mirror_c = run_df["x_mirror_c"].iloc[-1]
        end_y_mirror_c = run_df["y_mirror_c"].iloc[-1]

        coords = run_df[["x", "y"]].values
        run_length = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)) if coords.shape[0] >= 2 else 0.0

        mean_speed = run_df["speed"].mean()
        max_speed = run_df["speed"].max()

        dx = end_x_mirror_c - start_x_mirror_c
        dy = end_y_mirror_c - start_y_mirror_c
        run_angle_deg = np.degrees(np.arctan2(dy, dx))
        run_forward = dx > 0

        start_zone = get_zone(start_x_mirror_c, start_y_mirror_c, x_edges, y_edges)
        end_zone = get_zone(end_x_mirror_c, end_y_mirror_c, x_edges, y_edges)

        start_x_abs = run_df["x"].iloc[0]
        start_y_abs = run_df["y"].iloc[0]
        end_x_abs = run_df["x"].iloc[-1]
        end_y_abs = run_df["y"].iloc[-1]

        start_zone_abs = get_zone(start_x_abs, start_y_abs, x_edges, y_edges)
        end_zone_abs = get_zone(end_x_abs, end_y_abs, x_edges, y_edges)

        phase_of_play = run_df["phase_of_play"].iloc[0]
        in_possession = run_df["in_possession"].iloc[0]
        team_role = run_df["team_role"].iloc[0]
        position = run_df["position"].iloc[0]

        ball_carrier_df = extract_ball_carrier_df_fast(run_df, players_with_ball_df)

        overlapping = is_overlapping_run(run_df, ball_carrier_df, 2.0, 0.3, 60.0)
        underlapping = is_underlapping_run(run_df, ball_carrier_df, 2.0, 0.3, 150.0)
        is_diag = is_diagonal_run(run_df)

        runner_received_pass = runner_receives_pass(
            run_df,
            ball_carrier_df,
            events_df,
            players_df,
            match_id,
            metadata_dict,
            debug=False,
            debug_further=False
        )

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
            "tactical_diagonal": is_diag,
            "runner_received_pass": runner_received_pass
        })

    zones_df = pd.DataFrame(zone_records)
    return zones_df