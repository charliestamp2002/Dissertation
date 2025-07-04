import json
import gzip
import pathlib
import pandas as pd

def load_tracking_files(tracking_dir, n_files=None):
    TRACKING_DIR = pathlib.Path(tracking_dir)
    json_gz_paths = sorted(TRACKING_DIR.glob("tracking_*.json.gz"))
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
                        "x": px, "y": py, "z": pz,
                        "speed": p["speed"],
                    })

    frames_df = pd.DataFrame(frames)
    players_df = pd.DataFrame(players)

    return frames_df, players_df, used_match_ids