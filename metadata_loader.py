import pathlib
import json
import pandas as pd

def load_metadata_and_merge(players_df, metadata_dir_path, used_match_ids):
    metadata_dir = pathlib.Path(metadata_dir_path)
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

    used_match_suffixes = [match_id.split("_", 1)[1].replace(".json", "") for match_id in used_match_ids]

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

    players_df["match_id_clean"] = players_df["match_id"].str.replace(".json", "", regex=False)

    players_df = players_df.merge(
        meta_df,
        how="left",
        left_on=["match_id_clean", "optaId"],
        right_on=["match_id", "optaId"]
    )

    players_df.drop(columns=["match_id_clean", "match_id_y"], inplace=True, errors="ignore")
    players_df.rename(columns={"match_id_x": "match_id"}, inplace=True)

    return players_df