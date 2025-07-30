import gzip
import json
import numpy as np
import pandas as pd
import pathlib
import os
import joblib


def load_tracking_data(tracking_dir="tracking-compressed", n_files=3, block_size=250, num_blocks=50, seed=42):
    """
    Loads and samples tracking data from JSON.gz files.
    
    Args:
        tracking_dir (str or Path): Directory containing compressed tracking files.
        n_files (int): Number of files to load.
        block_size (int): Number of frames in each block.
        num_blocks (int): Number of blocks to sample per file.
        seed (int): Random seed for reproducibility.

    Returns:
        frames_df (pd.DataFrame): DataFrame containing frame-level ball info.
        players_df (pd.DataFrame): DataFrame containing player tracking data.
        used_match_ids (list): List of match IDs processed.
    """
    tracking_dir = pathlib.Path(tracking_dir)
    json_gz_paths = sorted(tracking_dir.glob("tracking_*.json.gz"))

    print(len(json_gz_paths), "tracking files found")

    frames = []
    players = []
    used_match_ids = []

    for json_gz_path in json_gz_paths[:n_files]:
        match_id = json_gz_path.stem
        used_match_ids.append(match_id)

        # First pass: count lines
        with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        max_possible_blocks = (total_lines - block_size) // block_size
        num_blocks_to_sample = min(num_blocks, max_possible_blocks)
        if num_blocks_to_sample <= 0:
            continue

        possible_starts = np.arange(0, total_lines - block_size + 1, block_size)
        rng = np.random.default_rng(seed)
        chosen_starts = rng.choice(possible_starts, size=num_blocks_to_sample, replace=False)
        sampled_indices = set()
        for start in chosen_starts:
            sampled_indices.update(range(start, start + block_size))

        # Second pass: read only selected frames
        with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i not in sampled_indices:
                    continue
                r = json.loads(line)

                # Frame-level data
                frames.append({
                    "match_id": match_id,
                    "period": r["period"],
                    "frameIdx": r["frameIdx"],
                    "gameClock": r["gameClock"],
                    "lastTouch_team": r["lastTouch"],
                    "ball_x": r["ball"]["xyz"][0],
                    "ball_y": r["ball"]["xyz"][1],
                    "ball_z": r["ball"]["xyz"][2],
                })

                # Player tracking data
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

def extract_event_metadata(events_path):
    """Extract match date and team names from a StatsBomb event file."""
    teams = set()
    match_date = None

    with open(events_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            obj = json.loads(line)
            if match_date is None and "match_date" in obj:
                match_date = obj["match_date"]
            if "team" in obj and obj["team"] is not None:
                teams.add(obj["team"]["name"])
            if "possession_team" in obj and obj["possession_team"] is not None:
                teams.add(obj["possession_team"]["name"])

    return {
        "events_file": str(events_path),
        "match_date": match_date,
        "team_names": list(teams),
    }

def load_event_metadata(events_dir="statsbomb_pl_data", used_match_ids=None):
    """Load metadata from StatsBomb event files, optionally filtered by used tracking matches."""
    events_dir = pathlib.Path(events_dir)
    event_files = sorted(events_dir.glob("*.json"))
    
    # If we have specific match IDs from tracking data, try to filter event files
    if used_match_ids is not None:
        print(f"Filtering event files to match {len(used_match_ids)} tracking matches...")
        # For now, load all and let the matching function handle the filtering
        # This could be optimized further if there's a clear mapping between
        # tracking file names and event file names
    
    metadata = [extract_event_metadata(path) for path in event_files]
    return pd.DataFrame(metadata)

def load_tracking_metadata(metadata_dir="metadata_SecondSpectrum"):
    """Load match metadata from Second Spectrum tracking files."""
    metadata_dir = pathlib.Path(metadata_dir)
    metadata_files = list(metadata_dir.glob("*.json"))

    records = []
    for path in metadata_files:
        # Try different encodings to handle various file formats
        encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
        meta = None
        
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    meta = json.load(f)
                break  # Success, exit the loop
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                continue  # Try next encoding
        
        if meta is None:
            print(f"Warning: Could not read {path} with any encoding, skipping...")
            continue

        if all(k in meta for k in ["year", "month", "day"]):
            match_date = f"{meta['year']:04}-{meta['month']:02}-{meta['day']:02}"
        else:
            match_date = None

        desc = meta.get("description", "")
        home_team, away_team = None, None
        if " - " in desc:
            teams_part = desc.split(":")[0].strip()
            home_team, away_team = teams_part.split(" - ")

        if path.stem.startswith("metadata_g"):
            suffix = path.stem.split("_")[1]
        else:
            suffix = path.stem.split("_")[0]

        records.append({
            "metadata_path": str(path),
            "tracking_suffix": suffix,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
        })

    return pd.DataFrame(records)

def match_events_to_tracking(event_meta_df, tracking_meta_df, team_code_map):
    """Match StatsBomb events to Second Spectrum tracking files using date + team."""
    tracking_meta_df = tracking_meta_df.copy()
    tracking_meta_df["home_team_full"] = tracking_meta_df["home_team"].map(team_code_map)
    tracking_meta_df["away_team_full"] = tracking_meta_df["away_team"].map(team_code_map)

    tracking_long = pd.concat([
        tracking_meta_df.assign(team_name=tracking_meta_df["home_team_full"]),
        tracking_meta_df.assign(team_name=tracking_meta_df["away_team_full"]),
    ])
    tracking_long["key"] = (
        tracking_long["match_date"].fillna("") + "_" +
        tracking_long["team_name"].fillna("")
    )

    events_exploded = event_meta_df.explode("team_names")
    events_exploded["team_name"] = events_exploded["team_names"].fillna("")
    events_exploded["key"] = (
        events_exploded["match_date"].fillna("") + "_" +
        events_exploded["team_name"].fillna("")
    )

    merged = events_exploded.merge(
        tracking_long,
        on="key",
        how="left",
        suffixes=("", "_tracking")
    )

    clean_df = (
        merged
        .dropna(subset=["tracking_suffix"])
        .drop_duplicates(subset=["events_file"])
        .reset_index(drop=True)
    )

    return clean_df

def add_event_timestamps(events_df):
    """Adds absolute and period-normalized seconds to StatsBomb event DataFrame."""
    events_df["minute"] = events_df["minute"].fillna(0).astype(int)
    events_df["second"] = events_df["second"].fillna(0).astype(int)
    #events_df["milliseconds"] = events_df.get("milliseconds", 0).fillna(0).astype(int)
    if "milliseconds" in events_df.columns:
        events_df["milliseconds"] = events_df["milliseconds"].fillna(0).astype(int)
    else:
        events_df["milliseconds"] = 0

    events_df["seconds_absolute"] = (
        events_df["minute"] * 60 +
        events_df["second"] +
        events_df["milliseconds"] / 1000
    )
    
    #events_df["seconds_period"] = events_df["seconds_absolute"]  # alias
    # Convert to continuous time (handling StatsBomb halftime reset)
    events_df["seconds_period"] = events_df.apply(
        lambda row: row["seconds_absolute"] if row["period"] == 1 
                   else row["seconds_absolute"] + 2700,  # Add 45 minutes for period 2
        axis=1
    )
    return events_df


def match_events_to_frames(events_df, frames_df_match):
    """
    Assigns the nearest frameIdx to each event by comparing game clock times.
    """
    period_to_frame_times = {
        period: group[["seconds_period", "frameIdx"]].sort_values("seconds_period")
        for period, group in frames_df_match.groupby("period")
    }

    assigned_frames, frame_diffs = [], []

    for _, e in events_df.iterrows():
        period = e["period"]
        seconds_event = e["seconds_period"]

        if period not in period_to_frame_times:
            assigned_frames.append(None)
            frame_diffs.append(None)
            continue

        times = period_to_frame_times[period]["seconds_period"].values
        frames = period_to_frame_times[period]["frameIdx"].values

        idx = np.argmin(np.abs(times - seconds_event))
        assigned_frames.append(frames[idx])
        frame_diffs.append(np.abs(times[idx] - seconds_event))

    events_df["frameIdx"] = assigned_frames
    events_df["frame_diff"] = frame_diffs
    return events_df


def find_events_during_run(run_df, events_df):
    """
    Filters events_df to those overlapping with the frame range of a given run.
    """
    period = run_df["period"].iloc[0]
    start_frame = run_df["frameIdx"].min()
    end_frame = run_df["frameIdx"].max()

    return events_df[
        (events_df["period"] == period) &
        (events_df["frameIdx"] >= start_frame) &
        (events_df["frameIdx"] <= end_frame)
    ]


def load_and_align_events(frames_df, players_df, event_tracking_df_clean, cache_path="events_dfs_by_match.joblib"):
    """
    Loads and aligns StatsBomb event data to tracking frames for valid matches.
    """
    # Normalize match_id format
    frames_df["match_id"] = frames_df["match_id"].str.replace(".json", "", regex=False)
    players_df["match_id"] = players_df["match_id"].str.replace(".json", "", regex=False)

    valid_tracking_ids = set(frames_df["match_id"].unique())
    filtered_df = event_tracking_df_clean[
        event_tracking_df_clean["tracking_suffix"].apply(lambda s: f"tracking_{s}").isin(valid_tracking_ids)
    ]

    if os.path.exists(cache_path):
        print(" Loaded precomputed events_dfs_by_match.")
        return joblib.load(cache_path)

    events_dfs_by_match = {}

    for _, row in filtered_df.iterrows():
        events_path = row["events_file"]
        tracking_suffix = row["tracking_suffix"]
        tracking_match_id = f"tracking_{tracking_suffix}"

        frames_df_match = frames_df[frames_df["match_id"] == tracking_match_id].copy()
        players_df_match = players_df[players_df["match_id"] == tracking_match_id]

        with open(events_path, "r", encoding="utf-8-sig") as f:
            event_rows = [json.loads(line) for line in f]
        events_df = pd.DataFrame(event_rows)
        events_df = add_event_timestamps(events_df)

        # Align frame clocks
        frames_df_match["seconds_period"] = frames_df_match.apply(
            lambda row: row["gameClock"] + (0 if row["period"] == 1 else 2700), axis=1
        )

        # Filter events per valid period and align with frames
        valid_periods = frames_df_match["period"].unique()
        filtered_events = []
        for period in valid_periods:
            f_period = frames_df_match[frames_df_match["period"] == period]
            e_period = events_df[events_df["period"] == period]
            if f_period.empty or e_period.empty:
                continue
            start_sec = f_period["seconds_period"].min()
            filtered_events.append(e_period[e_period["seconds_period"] >= start_sec])

        events_df = pd.concat(filtered_events, ignore_index=True)
        events_df = match_events_to_frames(events_df, frames_df_match)

        events_dfs_by_match[tracking_match_id] = events_df

    joblib.dump(events_dfs_by_match, cache_path)
    print(" Saved events_dfs_by_match to disk.")
    return events_dfs_by_match

def build_opta_metadata(players_df, used_match_ids, metadata_dir="metadata_SecondSpectrum"):
    """
    Loads Second Spectrum metadata files, maps player names/positions/optaIds to tracking data.

    Args:
        players_df (pd.DataFrame): DataFrame containing player tracking data.
        used_match_ids (list): List of tracking match IDs (e.g., "tracking_g2444470").
        metadata_dir (str or Path): Path to the Second Spectrum metadata directory.

    Returns:
        players_df (pd.DataFrame): Updated with player_name, position, team_role.
        metadata_df (pd.DataFrame): Player metadata lookup with optaId.
        metadata_dict (dict): Surname → optaId mapping for quick lookup.
    """
    metadata_dir = pathlib.Path(metadata_dir)
    all_metadata_files = list(metadata_dir.glob("*.json"))

    # Step 1: Extract suffixes from match IDs
    used_suffixes = [match_id.split("_", 1)[1].replace(".json", "") for match_id in used_match_ids]

    # Step 2: Build file map
    metadata_file_map = {}
    for path in all_metadata_files:
        fname = path.name
        if fname.startswith("metadata_g") and fname.endswith(".json"):
            suffix = fname.split("_")[1].split(".")[0]
        elif fname.endswith("_SecondSpectrum_Metadata.json"):
            suffix = fname.split("_")[0]
        else:
            continue
        metadata_file_map[suffix] = path

    # Step 3: Extract metadata for tracking suffixes
    opta_meta_lookup = {}
    meta_records = []

    for suffix in used_suffixes:
        metadata_path = metadata_file_map.get(suffix)
        if not metadata_path:
            print(f" No metadata found for match {suffix}")
            continue

        with open(metadata_path, "r", encoding="utf-8-sig") as f:
            meta = json.load(f)

        meta_records.append({
            "match_date": meta.get("matchDate"),
            "home_team": meta.get("homeTeamName"),
            "away_team": meta.get("awayTeamName"),
            "tracking_suffix": suffix,
            "metadata_path": str(metadata_path),
        })

        match_id = f"tracking_{suffix}"

        for side, team in [("homePlayers", "home"), ("awayPlayers", "away")]:
            for p in meta.get(side, []):
                key = (match_id, str(p["optaId"]))
                opta_meta_lookup[key] = {
                    "player_name": p.get("name"),
                    "position": p.get("position"),
                    "team_role": team,
                }

    print(f" Loaded metadata for {len(opta_meta_lookup)} players.")

    # Build DataFrame from lookup
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

    # Merge into players_df
    players_df = players_df.copy()
    players_df["match_id_clean"] = players_df["match_id"].str.replace(".json", "", regex=False)

    players_df = players_df.merge(
        meta_df,
        how="left",
        left_on=["match_id_clean", "optaId"],
        right_on=["match_id", "optaId"]
    )

    players_df.drop(columns=["match_id_clean", "match_id_y"], inplace=True, errors="ignore")
    players_df.rename(columns={"match_id_x": "match_id"}, inplace=True)

    # Build name→optaId dictionary using surnames
    def extract_surname(name):
        if pd.isna(name) or name is None:
            return ""
        name = name.lower().strip()
        parts = name.split()
        return parts[-1] if parts else ""

    metadata_df = meta_df.copy()
    metadata_df["surname"] = metadata_df["player_name"].apply(extract_surname)
    metadata_dict = dict(
        zip(metadata_df["surname"], metadata_df["optaId"].astype(str))
    )

    return players_df, metadata_df, metadata_dict

def extract_surname(name):
    if pd.isna(name) or name is None:
        return ""
    name = name.lower().strip()
    parts = name.split()
    return parts[-1] if parts else ""

def merge_assignments_with_metadata(assignments_df, final_runs_df, cluster_col):
    """
    Adds player metadata to each run in the cluster assignments.
    
    Args:
        assignments_df (pd.DataFrame): Contains 'run_id' and cluster column (e.g. 'ae_cluster').
        final_runs_df (pd.DataFrame): Original dataframe of all runs.
        cluster_col (str): Name of the column containing cluster assignments.

    Returns:
        pd.DataFrame: Merged dataframe with metadata and cluster assignment.
    """
    runs_meta_df = final_runs_df.groupby("run_id", as_index=False).first()
    merged_df = assignments_df.merge(runs_meta_df, on="run_id", how="left")

    # Drop duplicated columns if needed
    merged_df.drop(columns=[
        "position_y", "player_name_y", "team_role_y", "match_id_y", "playerId_y"
    ], inplace=True, errors="ignore")

    merged_df.rename(columns={
        "player_name_x": "player_name",
        "position_x": "position",
        "team_role_x": "team_role",
        "match_id_x": "match_id",
        "playerId_x": "playerId",
    }, inplace=True)

    return merged_df

def add_position_buckets(df, position_col="position"):
    """
    Adds a new column 'position_bucket' mapping fine-grained positions to broad roles.
    """
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

    df["position_bucket"] = df[position_col].map(lambda pos: position_bucket_map.get(pos, "unknown"))
    return df

def compute_position_pivot(df, cluster_col="ae_cluster"):
    """
    Returns pivot table: [cluster x position]
    """
    position_counts = (
        df.groupby([cluster_col, "position"])
        .size()
        .reset_index(name="num_runs")
        .sort_values([cluster_col, "num_runs"], ascending=[True, False])
    )

    pivot = (
        position_counts
        .pivot_table(index=cluster_col, columns="position", values="num_runs", fill_value=0)
        .reset_index()
    )
    return pivot


def compute_bucket_pivot(df, cluster_col="ae_cluster"):
    """
    Returns pivot table: [cluster x position_bucket]
    """
    bucket_counts = (
        df.groupby([cluster_col, "position_bucket"])
        .size()
        .reset_index(name="num_runs")
        .sort_values([cluster_col, "num_runs"], ascending=[True, False])
    )

    pivot = (
        bucket_counts
        .pivot_table(index=cluster_col, columns="position_bucket", values="num_runs", fill_value=0)
        .reset_index()
    )
    return pivot

def get_valid_resampled_tensor(runs_df, num_points=25, return_indices=False):
    """
    Filters and resamples trajectories to a fixed number of points (e.g., 25).
    Only includes runs with at least `num_points` original frames.
    Returns a NumPy array of shape [N, num_points, 2].
    """
    from run_segmentation import resample_coords

    traj_tensor = []
    valid_run_ids = []
    skipped = 0

    for run_id in runs_df["run_id"].unique():
        run = runs_df[runs_df["run_id"] == run_id]
        coords = run[["x_mirror_c", "y_mirror_c"]].values

        if coords.shape[0] < num_points:
            skipped += 1
            continue

        try:
            resampled = resample_coords(coords, num_points=num_points)
            if resampled.shape == (num_points, 2):
                traj_tensor.append(resampled)
                valid_run_ids.append(run_id)
            else:
                skipped += 1
        except Exception as e:
            print(f"Error resampling run_id {run_id}: {e}")
            skipped += 1

    print(f"Total valid runs: {len(traj_tensor)}")
    print(f"Total skipped runs: {skipped}")
    
    stacked = np.stack(traj_tensor)
    if return_indices:
        return stacked, valid_run_ids
    else:
        return stacked


### –––––––––––––––––––––– OLD CODE –––––––––––––––––––––––– ###

