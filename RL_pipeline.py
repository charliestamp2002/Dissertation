#!/usr/bin/env python3
"""
RL Pipeline for Off-Ball Run Valuation

This script implements a complete RL pipeline for evaluating off-ball runs
using clustering assignments and various reward functions (C-OBSO, Pitch Control, OBV).
"""

import pandas as pd
import numpy as np
import joblib
import torch
from tqdm import tqdm

# Import custom modules
from RL_networks import GVRNN, FootballOffBallRL, build_tracking_dataset, train_gvrnn
from RL_state_funcs import c_obso_state_fn, simple_state_fn
from RL_reward_funcs import c_obso_reward_fn, pitch_control_reward_fn, obv_reward_fn, add_velocities_to_players_df

def ensure_seconds_period(final_runs_df, frames_df):
    """
    Ensures seconds_period is present in final_runs_df.
    """
    frames_df["seconds_period"] = frames_df.apply(
            lambda row: row["gameClock"] + (0 if row["period"] == 1 else 2700), axis=1
        )
    if "seconds_period" not in final_runs_df.columns:
        final_runs_df = final_runs_df.merge(
            frames_df[["match_id", "period", "frameIdx", "seconds_period"]],
            on=["match_id", "period", "frameIdx"],
            how="left",
        )
    return final_runs_df

def load_data():
    """
    Load all required data for the RL pipeline.
    """
    print("Loading data files...")
    
    # Load main dataframes
    final_runs_df = pd.read_parquet("outputs/final_runs_df.parquet")
    players_df = pd.read_parquet("outputs/players_df.parquet")
    frames_df = pd.read_parquet("outputs/frames_df.parquet")
    
    # Load assignments from one of the clustering methods (using bezier as default)
    assignments_df = pd.read_pickle("outputs/assignments_zones_bezier.pkl")
    
    events_dfs_by_match = joblib.load("events_dfs_by_match.joblib")
    
    print(f"Loaded {len(final_runs_df)} runs, {len(players_df)} player records")
    print(f"Loaded {len(frames_df)} frame records")
    print(f"Loaded {len(assignments_df)} cluster assignments")
    
    # Add seconds_period to final_runs_df for OBV reward calculation
    print("Adding seconds_period to runs data...")
    final_runs_df = ensure_seconds_period(final_runs_df, frames_df)
    print("✅ seconds_period added successfully")
    
    return final_runs_df, players_df, frames_df, assignments_df, events_dfs_by_match

def prepare_data(final_runs_df, players_df, assignments_df):
    """
    Prepare data by adding velocities and merging cluster assignments.
    """
    print("Preparing data...")
    
    # Copy and add velocities
    runs_df = final_runs_df.copy()
    players_df = add_velocities_to_players_df(players_df, fps=25)
    
    # Check columns before merge
    print("Before merge:", runs_df.columns.tolist())
    print("Assignments columns:", assignments_df.columns.tolist())
    
    # Merge in assigned_cluster
    runs_df = runs_df.merge(
        assignments_df[["run_id", "assigned_cluster"]],
        on="run_id",
        how="left"
    )
    
    print("After merge:", runs_df.columns.tolist())
    print(f"Runs with cluster assignments: {runs_df['assigned_cluster'].notna().sum()}")
    
    return runs_df, players_df

def create_rl_model(method="dqn", reward_func="c-obso", state_dim=90, action_space_size=70):
    """
    Create and configure the RL model with GVRNN.
    """
    print(f"Creating RL model with method={method}, reward_func={reward_func}")
    
    # Define GVRNN
    gvrnn_model = GVRNN(
        state_dim=state_dim,
        action_space_size=action_space_size,
        latent_dim=16,
        hidden_dim=128,
        future_steps=20
    )
    
    # Select reward function
    if reward_func == "c-obso":
        reward_fn = c_obso_reward_fn
        state_fn = c_obso_state_fn
    elif reward_func == "pc":
        reward_fn = pitch_control_reward_fn
        state_fn = c_obso_state_fn  # PC also needs velocities
    elif reward_func == "obv":
        reward_fn = obv_reward_fn
        state_fn = simple_state_fn
    else:
        raise ValueError(f"Unknown reward function: {reward_func}")
    
    # Create RL model
    model = FootballOffBallRL(
        method=method,
        state_dim=state_dim,
        action_space_size=action_space_size,
        reward_func=reward_func,
        state_fn=state_fn,
        reward_fn=reward_fn,
        gvrnn_model=gvrnn_model,
        n_iters=20,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return model, gvrnn_model

def main():
    """
    Main RL pipeline execution.
    """
    print("Starting RL Pipeline for Off-Ball Run Valuation")
    print("=" * 50)
    
    # 1. Load data
    final_runs_df, players_df, frames_df, assignments_df, events_dfs_by_match = load_data()
    
    # 2. Prepare data
    runs_df, players_df = prepare_data(final_runs_df, players_df, assignments_df)
    
    # 3. Create RL model
    model, gvrnn_model = create_rl_model(
        method="dqn",
        reward_func="c-obso",
        state_dim=90,
        action_space_size=70
    )
    
    # 4. Prepare RL dataset
    print("\nPreparing RL dataset...")
    rl_data = model.prepare_data(
        runs_df=runs_df,
        players_df=players_df,
        frames_df=frames_df,
        assignments_df=assignments_df,
        events_dfs_by_match=events_dfs_by_match
    )
    
    # 5. Train RL model
    print("\nTraining RL model...")
    model.fit()
    
    # 6. Evaluate and rank players
    print("\nRanking players by Q-values...")
    player_rankings = model.rank_players()
    print("\nTop 10 players by Q-value:")
    print(player_rankings.head(10))
    
    # 7. Optional: Train GVRNN for counterfactual analysis
    print("\nBuilding GVRNN dataset for counterfactual analysis...")
    gvrnn_dataset = build_tracking_dataset(
        runs_df=runs_df,
        assignments_df=assignments_df,
        players_df=players_df,
        frames_df=frames_df,
        model=model,
        T_past=10,
        T_future=10
    )
    
    if len(gvrnn_dataset) > 0:
        print("Training GVRNN...")
        train_gvrnn(
            gvrnn_model=gvrnn_model,
            dataset=gvrnn_dataset,
            num_epochs=20,
            batch_size=32
        )
        print("GVRNN training completed.")
    
    print("\nRL Pipeline completed successfully!")
    return model, player_rankings

if __name__ == "__main__":
    try:
        model, rankings = main()
    except Exception as e:
        print(f"Error in RL pipeline: {e}")
        import traceback
        traceback.print_exc()