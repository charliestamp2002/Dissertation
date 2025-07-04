from data_utils import *
from features import *
from bezier_clustering import *
from plotting import plot_all_cluster_trajectories_on_pitch

TRACKING_DIR = pathlib.Path("tracking-compressed")
METADATA_DIR = pathlib.Path("metadata_SecondSpectrum")

frames_df, players_df, used_match_ids = load_tracking_data(TRACKING_DIR, n_files=3)
meta_df = load_metadata(METADATA_DIR, used_match_ids)
players_df = merge_player_metadata(players_df, meta_df) # Not Sure ??? Check...

runs_list = segment_runs(players_df)
runs_list = filter_off_ball_runs_with_distance(runs_list, frames_df, players_df, min_distance=3.0)
all_runs_df = annotate_runs_with_metadata(runs_list)
all_runs_df = all_runs_df.groupby("run_id", group_keys=False).apply(mirror_group)
centroid_dict = compute_team_centroids(players_df)
frame_last_touch_team, player_side_lookup = build_frame_lookups(frames_df, players_df)

final_runs_df = adjust_runs(all_runs_df, centroid_dict, frame_last_touch_team)


players_with_ball_df = players_df.merge(
    frames_df[["match_id", "period", "frameIdx", "ball_x", "ball_y", "lastTouch_team"]],
    on=["match_id", "period", "frameIdx"],
    how="left",
    suffixes=("", "_ball"))

zones_df = compute_run_features(final_runs_df, players_with_ball_df)

cluster_control_points, assignments_df = bezier_kmeans_clustering(
    final_runs_df,
    k_clusters=70,
    num_control_points=4,
    num_points=50,
    max_iterations=10,
    tolerance=1e-3,
    random_seed=42)

assignments_zones = merge_assignments_with_zones(assignments_df, zones_df)
position_pivot, bucket_pivot = compute_position_buckets(assignments_zones)
merged_df = merge_assignments_with_run_metadata(assignments_df, final_runs_df)

plot_all_cluster_trajectories_on_pitch(
    final_runs_df,
    assignments_zones,
    cluster_control_points,
    bucket_pivot=bucket_pivot,
    num_control_points=4,
    max_runs_per_cluster=200,
    plot_absolute_positions=True,
    start_zones=None,
    end_zones=None,  # You can specify zones like [1, 2, 3] if you want to filter
    phases_of_play=None,
    positions=None,
    use_absolute_zones=True, # True or False are options
    start_zones_absolute=None,
    end_zones_absolute=None,
    run_angle_range=None,  # Example: filter for runs with angles between 0 and 180 degrees
    run_forward=True,
    run_length_range = None,  # Example: filter for runs with length between 0 and 100 meters
    mean_speed_range = None,  # Example: filter for runs with mean speed between 0 and 8 m/s
    max_speed_range = None,
    tactical_overlap=None,
    tactical_underlap=None,
    tactical_diagonal=True)


if __name__ == "__main__":
    # print("FRAMES DF", frames_df.head())
    # print("PLAYERS DF", players_df.head())
    # print("META DF", meta_df.head())
    #print("META DF MERGED", meta_df_merged.head())
    #print(all_runs_df.head())
    #print(centroid_dict)
    print("FINAL RUNS DF", final_runs_df.columns.to_list())

