import json
import gzip
import numpy as np
import pandas as pd
import pathlib

def load_tracking_data(tracking_dir, n_files=None):
    """
    Load compressed JSON tracking data from tracking_dir.

    Returns:
        frames_df
        players_df
        used_match_ids
    """
    json_gz_paths = sorted(tracking_dir.glob("tracking_*.json.gz"))

    if n_files is None:
        n_files = len(json_gz_paths)

    frames = []
    players = []
    used_match_ids = []

    for json_gz_path in json_gz_paths[:n_files]:
        match_id = json_gz_path.stem
        used_match_ids.append(match_id)

        records = []
        with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        for r in records:
            f_data = {
                "match_id": match_id,
                "period": r["period"],
                "frameIdx": r["frameIdx"],
                "gameClock": r["gameClock"],
                "lastTouch_team": r["lastTouch"],
                "ball_x": r["ball"]["xyz"][0],
                "ball_y": r["ball"]["xyz"][1],
                "ball_z": r["ball"]["xyz"][2],
            }
            frames.append(f_data)

            for side in ["homePlayers", "awayPlayers"]:
                for p in r[side]:
                    px, py, pz = p["xyz"]
                    players.append({
                        "match_id": match_id,
                        "period": r["period"],
                        "frameIdx": r["frameIdx"],
                        "side": "home" if side == "homePlayers" else "away",
                        "playerId": p["playerId"],
                        "optaId": str(p["optaId"]),
                        "number": p["number"],
                        "x": px,
                        "y": py,
                        "z": pz,
                        "speed": p["speed"],
                    })

    frames_df = pd.DataFrame(frames)
    players_df = pd.DataFrame(players)

    return frames_df, players_df, used_match_ids


def load_metadata(metadata_dir, used_match_ids):
    """
    Loads Second Spectrum metadata JSONs.

    Returns:
        meta_df
    """
    used_match_suffixes = [match_id.split("_", 1)[1].replace(".json", "") for match_id in used_match_ids]

    all_metadata_files = list(metadata_dir.glob("*.json"))

    metadata_file_map = {}
    for path in all_metadata_files:
        filename = path.name
        if filename.startswith("metadata_g") and filename.endswith(".json"):
            suffix = filename.split("_")[1].split(".")[0]
        elif filename.endswith("_SecondSpectrum_Metadata.json"):
            suffix = filename.split("_")[0]
        else:
            continue
        metadata_file_map[suffix] = path

    opta_meta_lookup = {}

    for suffix in used_match_suffixes:
        metadata_path = metadata_file_map.get(suffix)
        if not metadata_path:
            print(f"No metadata found for match {suffix}")
            continue

        with open(metadata_path, "r", encoding="utf-8-sig") as f:
            meta = json.load(f)

        match_id = f"tracking_{suffix}"

        for side, team in [("homePlayers", "home"), ("awayPlayers", "away")]:
            for p in meta.get(side, []):
                key = (match_id, str(p["optaId"]))
                opta_meta_lookup[key] = {
                    "player_name": p.get("name"),
                    "position": p.get("position"),
                    "team_role": team,
                }

    meta_df = pd.DataFrame([
        {
            "match_id": match_id,
            "optaId": opta_id,
            "player_name": info["player_name"],
            "position": info["position"],
            "team_role": info["team_role"],
        }
        for (match_id, opta_id), info in opta_meta_lookup.items()
    ])

    return meta_df


def merge_player_metadata(players_df, meta_df):
    """
    Merges player tracking DataFrame with metadata DataFrame.
    """
    players_df["match_id_clean"] = players_df["match_id"].str.replace(".json", "", regex=False)

    merged = players_df.merge(
        meta_df,
        how="left",
        left_on=["match_id_clean", "optaId"],
        right_on=["match_id", "optaId"]
    )

    merged.drop(columns=["match_id_clean", "match_id_y"], inplace=True, errors="ignore")
    merged.rename(columns={"match_id_x": "match_id"}, inplace=True)

    return merged


def segment_runs(players_df, speed_threshold=2.0):
    """
    Segments continuous runs for each player where speed > threshold.
    """
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


def filter_off_ball_runs_with_distance(runs_list, frames_df, players_df, min_distance=3.0):
    """
    Filters runs:
    - player never touched ball
    - always at least `min_distance` from the ball
    """
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

def annotate_runs_with_metadata(runs_list):
    # Annotate each run with player metadata
    annotated_runs = []

    for run_df in runs_list:
        # Make a copy of the run to avoid modifying in-place
        run_df = run_df.copy()

        # Extract metadata from the first row (same for entire run)
        meta_fields = ["playerId", "optaId", "match_id", "player_name", "position", "team_role"]
        for field in meta_fields:
            run_df[field] = run_df.iloc[0][field]

        annotated_runs.append(run_df)

    # Assign a unique run_id to each run
    for i, run_df in enumerate(annotated_runs):
        run_df["run_id"] = i

    # Optional: Combine into one dataframe
    all_runs_df = pd.concat(annotated_runs, ignore_index=True)

    return all_runs_df


def mirror_group(group):
    y_mean = group["y"].mean()
    if y_mean < 0:
        group["y_mirror"] = -group["y"]
    else:
        group["y_mirror"] = group["y"]
    group["x_mirror"] = group["x"]
    return group


def should_flip_x(team_role, period):
    if period == 1:
        return team_role == "away"
    elif period == 2:
        return team_role == "home"
    else:
        return False


def compute_team_centroids(players_df):
    centroid_dict = {}
    players_df["number"] = players_df["number"].astype(int)

    for (match_id, period, frame_idx), group in players_df.groupby(["match_id", "period", "frameIdx"]):
        for side in ["home", "away"]:
            team_players = group[(group["side"] == side) & (group["number"] != 1)]
            if not team_players.empty:
                centroid = team_players[["x", "y"]].mean().values
            else:
                centroid = np.array([0.0, 0.0])
            centroid_dict[(match_id, period, frame_idx, side)] = centroid

    return centroid_dict


def build_frame_lookups(frames_df, players_df):
    frame_last_touch_team = frames_df.set_index(["match_id", "period", "frameIdx"])["lastTouch_team"].to_dict()
    player_side_lookup = players_df.set_index(["match_id", "period", "frameIdx", "playerId"])["side"].to_dict()
    return frame_last_touch_team, player_side_lookup


def adjust_runs(all_runs_df, centroid_dict, frame_last_touch_team):
    adjusted_runs_list = []

    grouped = all_runs_df.groupby(["match_id", "period", "playerId", "run_id"], group_keys=False)

    for _, run_df in grouped:
        run_df = run_df.sort_values("frameIdx")
        match_id = run_df["match_id"].iloc[0]
        period = run_df["period"].iloc[0]
        start_frame = run_df["frameIdx"].iloc[0]

        key = (match_id, period, start_frame)
        possession_side = frame_last_touch_team.get(key)

        team_role = run_df["team_role"].iloc[0]

        if possession_side is None:
            in_possession = np.nan
            phase_of_play = np.nan
        else:
            in_possession = (team_role == possession_side)
            phase_of_play = "attack" if in_possession else "defend"

        run_df["in_possession"] = in_possession
        run_df["phase_of_play"] = phase_of_play

        team_centroid = centroid_dict.get(
            (match_id, period, start_frame, team_role),
            np.array([0.0, 0.0])
        )

        run_df["x_c"] = run_df["x"] - team_centroid[0]
        run_df["y_c"] = run_df["y"] - team_centroid[1]
        run_df["x_mirror_c"] = run_df["x_mirror"] - team_centroid[0]
        run_df["y_mirror_c"] = run_df["y_mirror"] - team_centroid[1]

        flip_x = should_flip_x(team_role, period)
        if flip_x:
            run_df["x_mirror"] = -run_df["x_mirror"]
            run_df["x_mirror_c"] = -run_df["x_mirror_c"]

        adjusted_runs_list.append(run_df)

    final_runs_df = pd.concat(adjusted_runs_list, ignore_index=True)

    return final_runs_df

def merge_assignments_with_zones(assignments_df, zones_df):
    assignments_zones = assignments_df.merge(
        zones_df,
        on="run_id",
        how="left"
    )

    # Clean up duplicates if merge produced suffixes
    assignments_zones.drop(columns=[
        "position_y",
        "team_role_y"
    ], inplace=True, errors="ignore")

    assignments_zones.rename(columns={
        "position_x": "position",
        "team_role_x": "team_role",
    }, inplace=True)

    return assignments_zones


def compute_position_buckets(assignments_zones):
    # Group and count runs per position
    position_detail_counts = (
        assignments_zones
        .groupby(["assigned_cluster", "position"])
        .size()
        .reset_index(name="num_runs")
        .sort_values(["assigned_cluster", "num_runs"], ascending=[True, False])
    )

    position_pivot = (
        position_detail_counts
        .pivot_table(index="assigned_cluster",
                     columns="position",
                     values="num_runs",
                     fill_value=0)
        .reset_index()
    )

    # Mapping of fine-grained positions → high-level buckets
    position_bucket_map = {
        "GK": "sub", "SUB": "sub",
        "CB": "defender", "RCB": "defender", "LCB": "defender",
        "RB": "defender", "LB": "defender", "RWB": "defender", "LWB": "defender",
        "CDM": "midfielder", "RDM": "midfielder", "LDM": "midfielder",
        "CM": "midfielder", "RCM": "midfielder", "LCM": "midfielder",
        "CAM": "midfielder", "RM": "midfielder", "LM": "midfielder",
        "LW": "attacker", "RW": "attacker", "ST": "attacker",
        "CF": "attacker", "RF": "attacker", "LF": "attacker",
    }

    assignments_zones["position_bucket"] = assignments_zones["position"].map(
        lambda pos: position_bucket_map.get(pos, "unknown")
    )

    bucket_counts = (
        assignments_zones
        .groupby(["assigned_cluster", "position_bucket"])
        .size()
        .reset_index(name="num_runs")
        .sort_values(["assigned_cluster", "num_runs"], ascending=[True, False])
    )

    bucket_pivot = (
        bucket_counts
        .pivot_table(index="assigned_cluster",
                     columns="position_bucket",
                     values="num_runs",
                     fill_value=0)
        .reset_index()
    )

    return position_pivot, bucket_pivot

def merge_assignments_with_run_metadata(assignments_df, final_runs_df):
    runs_meta_df = final_runs_df.groupby("run_id", as_index=False).first()

    merged_df = assignments_df.merge(
        runs_meta_df,
        on="run_id",
        how="left"
    )

    merged_df.drop(columns=[
        "position_y",
        "player_name_y",
        "team_role_y",
        "match_id_y",
        "playerId_y",
    ], inplace=True, errors="ignore")

    merged_df.rename(columns={
        "player_name_x": "player_name",
        "position_x": "position",
        "team_role_x": "team_role",
        "match_id_x": "match_id",
        "playerId_x": "playerId",
    }, inplace=True)

    return merged_df

