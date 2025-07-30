import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from tqdm import tqdm
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator


import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal
import torch.distributions as D
from torch.utils.data import TensorDataset, DataLoader, Dataset


# --------------------------------------------------------------------------------
# C-OBSO REWARD FUNCTION
# --------------------------------------------------------------------------------

def compute_transition_field(state_row, grid_size=5.0, sigma=14.0):
    """
    P(T_r | D): Gaussian centered on ball position → where next on-ball event occurs.
    """
    ball_x, ball_y = state_row["ball_coords"]
    xs = np.arange(0, 105 + grid_size, grid_size) # Need to check consistent with tracking data
    ys = np.arange(0, 68  + grid_size, grid_size) # Need to check consistent with tracking data
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.dstack([X, Y])
    rv = multivariate_normal(mean=[ball_x, ball_y],
                             cov=[[sigma**2, 0], [0, sigma**2]])
    return rv.pdf(pos)  # shape (nx, ny)


def compute_PPCF(players_df, match_id, period, frameIdx,
                 lambdas=None, s=0.45, grid_size=5.0):
    """
    P(C_r | D): Potential Pitch Control Field (PPCF).
    We discretize and integrate the Poisson‐arrival model.
    For simplicity we approximate final control probability at T→∞:
       P_control(r) = 1 - exp( - λ_j * f_j(r) )  summed over j
    where f_j(r) = sigmoid( (T_exp - T)/sqrt(3 s / π) ), approximated here by
    a logistic of distance to r.
    """
    # load the 22 players at that frame
    frame = players_df.query(
        "match_id==@match_id & period==@period & frameIdx==@frameIdx"
    ).reset_index(drop=True)
    coords = frame[["x","y"]].values  # (22,2)
    # default λ_j = same for all
    if lambdas is None:
        lambdas = np.ones(len(coords)) * 4.3

    xs = np.arange(0, 105 + grid_size, grid_size)
    ys = np.arange(0, 68  + grid_size, grid_size)
    PPCF = np.zeros((len(xs), len(ys)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            dists = np.linalg.norm(coords - np.array([x, y]), axis=1)
            # expected intercept time T_exp ∝ dist / avg_speed (≈5m/s)
            Texp = dists / 5.0
            # logistic f_j
            f_j = 1.0 / (1.0 + np.exp(-(Texp - Texp) / np.sqrt(3*s/np.pi)))
            # per‐player asymptotic control: 1 - exp(-λ_j * f_j)
            p_control = 1 - np.exp(-lambdas * f_j)
            # assuming independence, team control = 1 - ∏(1 - p_j)
            PPCF[i,j] = 1 - np.prod(1 - p_control)
    return PPCF


def compute_shot_field(players_df, frames_df,
                       match_id, period, frameIdx,
                       grid_size=5.0, C=1.0, c=0.2):
    """
    P(S_r | D): shot success probability from each grid cell,
    adjusted by defenders' shot‐blocking distribution.
    """
    # load defenders and their positions
    frame = players_df.query(
        "match_id==@match_id & period==@period & frameIdx==@frameIdx"
    ).reset_index(drop=True)
    coords = frame[["x","y"]].values
    sides  = frame["side"].values  # "home" or "away"
    # determine attacking side
    last = frames_df.query(
        "match_id==@match_id & period==@period & frameIdx==@frameIdx"
    )["lastTouch_team"].iat[0]
    attack_mask = (sides == last)

    xs = np.arange(0, 105 + grid_size, grid_size)
    ys = np.arange(0, 68  + grid_size, grid_size)
    SF = np.zeros((len(xs), len(ys)))

    # goal is at (105, 34)
    goal = np.array([105.0, 34.0])
    #CHECK IF THESE COORDS ARE CORRECT ACCORDING TO TRACKING FILES

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            r = np.array([x, y])
            # generate shot vectors: here just one direct vector
            vec = goal - r
            # defenders on attack side
            def_idxs = np.where(~attack_mask)[0]
            # compute block value: sum of Gaussians along that vector
            V_block = 0
            for d in def_idxs:
                dpos = coords[d]
                # project defender onto line ‖vec‖
                t = np.dot(dpos - r, vec) / np.dot(vec, vec)
                proj = r + t * vec
                # distance from line
                perp = np.linalg.norm(dpos - proj)
                sigma2 = 0.5 + np.linalg.norm(r - dpos)
                V_block += multivariate_normal.pdf(
                    [perp, 0], mean=[0,0], cov=[[sigma2,0],[0,sigma2]]
                )
            V_shot = max(0.0, C*(c - V_block))
            SF[i,j] = V_shot

    # normalize to [0,1]
    SF = SF / (SF.max() + 1e-8)
    return SF


def compute_OBSO(state_row, players_df, frames_df,
                 grid_size=5.0, sigma=14.0):
    """
    Full OBSO value:
      V_OBSO = Σ_r P(T_r) · P(C_r) · P(S_r)
    """
    # 1) transition
    T = compute_transition_field(state_row, grid_size, sigma)
    # 2) control
    # we need match_id,period,frameIdx from state_row:
    m, p, f = state_row["match_id"], state_row["period"], state_row["frameIdx"]
    C = compute_PPCF(players_df, m, p, f, grid_size=grid_size)
    # 3) shot
    S = compute_shot_field(players_df, frames_df, m, p, f, grid_size=grid_size)
    # combine
    return np.sum(T * C * S) * (grid_size**2)

def c_obso_reward_fn(run_df, players_df, frames_df):
    """
    Computes C-OBSO reward for the run.

    Parameters
    ----------
    run_df : pd.DataFrame
        DataFrame with the trajectory of a single run.

    players_df : pd.DataFrame
        Players tracking data.

    frames_df : pd.DataFrame
        Frames tracking data.

    Returns
    -------
    float
        C-OBSO value for this run.
    """
    end_row = run_df.iloc[-1]
    match_id = end_row["match_id"]
    period = end_row["period"]
    frameIdx = end_row["frameIdx"]

    # Extract state for the end frame
    # This matches your class logic:
    state_row = {
        "match_id": match_id,
        "period": period,
        "frameIdx": frameIdx
    }

    # Add ball coords:
    ball_at_frame = frames_df[
        (frames_df["match_id"] == match_id) &
        (frames_df["period"] == period) &
        (frames_df["frameIdx"] == frameIdx)
    ]
    if ball_at_frame.empty:
        return 0.0

    ball_x = ball_at_frame["ball_x"].iloc[0]
    ball_y = ball_at_frame["ball_y"].iloc[0]
    state_row["ball_coords"] = np.array([ball_x, ball_y])

    return compute_OBSO(
        state_row,
        players_df,
        frames_df
    )


def add_velocities_to_players_df(players_df, fps=25):
    """
    Computes forward-difference player velocities and adds vx, vy
    to the players_df DataFrame.

    Parameters
    ----------
    players_df : pd.DataFrame
        Must contain:
            - match_id
            - period
            - frameIdx
            - playerId
            - x
            - y

    fps : float
        Tracking data frame rate (Hz).

    Returns
    -------
    players_df : pd.DataFrame
        DataFrame with new columns vx, vy.
    """
    dt = 1.0 / fps

    # Ensure proper sort
    players_df_sorted = players_df.sort_values(
        ["match_id", "period", "playerId", "frameIdx"]
    ).copy()

    # Forward differences
    players_df_sorted["x_next"] = players_df_sorted.groupby(
        ["match_id", "period", "playerId"]
    )["x"].shift(-1)

    players_df_sorted["y_next"] = players_df_sorted.groupby(
        ["match_id", "period", "playerId"]
    )["y"].shift(-1)

    players_df_sorted["vx"] = (
        (players_df_sorted["x_next"] - players_df_sorted["x"]) / dt
    )
    players_df_sorted["vy"] = (
        (players_df_sorted["y_next"] - players_df_sorted["y"]) / dt
    )

    # For last frame of each player’s data, set velocity to zero
    players_df_sorted["vx"] = players_df_sorted["vx"].fillna(0.0)
    players_df_sorted["vy"] = players_df_sorted["vy"].fillna(0.0)

    # Drop helper columns
    players_df_sorted = players_df_sorted.drop(columns=["x_next", "y_next"])

    return players_df_sorted



def compute_pitch_control(state_row, players_df, attacking_side, match_id, period, frameIdx, grid_size=5.0):
    """
    Computes pitch control given a frame and dynamic team assignment.

    Parameters
    ----------
    state_row : dict
        Contains:
            - player_coords : (44,)
            - player_velocities : (44,)
            - ball_coords : (2,)

    players_df : pd.DataFrame
        Player tracking data.

    attacking_side : str
        Either "home" or "away".

    match_id, period, frameIdx : str, int, int
        Frame identifiers.

    grid_size : float
        Cell size in meters.

    Returns
    -------
    float
        Total controlled area (in m²) with >50% pitch control.
    """
    player_xy = state_row["player_coords"].reshape(-1, 2)   # (22, 2)
    player_vxy = state_row["player_velocities"].reshape(-1, 2)

    # Find indices of attacking players
    frame_players = players_df[
        (players_df["match_id"] == match_id) &
        (players_df["period"] == period) &
        (players_df["frameIdx"] == frameIdx)
    ].reset_index(drop=True)

    team_mask = frame_players["side"] == attacking_side
    team_idx = np.where(team_mask.values)[0]

    grid_x = np.arange(0, 105 + grid_size, grid_size) # Check grid is compatible 
    grid_y = np.arange(0, 68 + grid_size, grid_size)

    pc_map = np.zeros((len(grid_x), len(grid_y)))

    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            t_arrival = []
            for pos, vel in zip(player_xy, player_vxy):
                dist = np.linalg.norm([x, y] - pos)
                speed = np.linalg.norm(vel)
                speed = max(speed, 1.0)  # avoid divide-by-zero
                t_arrival.append(dist / speed)

            t_arrival = np.array(t_arrival)

            t_team = t_arrival[team_idx]
            t_opp = np.delete(t_arrival, team_idx)

            p_team = np.exp(-np.min(t_team))
            p_opp = np.exp(-np.min(t_opp))

            pc = p_team / (p_team + p_opp + 1e-8)

            pc_map[i, j] = pc

    area_controlled = np.sum(pc_map > 0.5) * (grid_size ** 2)
    return area_controlled

def compute_pc_without_runner(state_row, runner_idx):
    # Remove runner from state
    player_xy = state_row["player_coords"].reshape(-1, 2)
    player_vxy = state_row["player_velocities"].reshape(-1, 2)

    # Remove runner
    player_xy_no_run = np.delete(player_xy, runner_idx, axis=0)
    player_vxy_no_run = np.delete(player_vxy, runner_idx, axis=0)

    state_no_run = {
        "player_coords": player_xy_no_run.flatten(),
        "player_velocities": player_vxy_no_run.flatten(),
        "ball_coords": state_row["ball_coords"]
    }

    return compute_pitch_control(state_no_run)

def find_runner_idx_from_players_df(players_df, match_id, period, frameIdx, player_id):
    """
    Finds the runner's index among 22 players in a single frame.

    Returns
    -------
    runner_idx : int
        Integer between 0-21
    """
    frame_players = players_df[
        (players_df["match_id"] == match_id) &
        (players_df["period"] == period) &
        (players_df["frameIdx"] == frameIdx)
    ].reset_index(drop=True)

    mask = frame_players["playerId"] == player_id
    if not mask.any():
        raise ValueError(f"Player {player_id} not found in frame {frameIdx}")
    
    runner_idx = mask.idxmax()
    return runner_idx


def compute_pc_with_runner_still(state_row, runner_idx, players_df, attacking_side, grid_size=5.0):
    player_xy = state_row["player_coords"].reshape(-1, 2)
    player_vxy = state_row["player_velocities"].reshape(-1, 2)

    # Freeze runner
    player_vxy[runner_idx, :] = 0.0

    state_counterfactual = {
        "player_coords": player_xy.flatten(),
        "player_velocities": player_vxy.flatten(),
        "ball_coords": state_row["ball_coords"]
    }

    return compute_pitch_control(
        state_counterfactual,
        players_df,
        attacking_side,
        state_row["match_id"],
        state_row["period"],
        state_row["frameIdx"],
        grid_size=grid_size
    )

def get_attacking_side(frames_df, match_id, period, frameIdx):
    """
    Returns 'home' or 'away' indicating the attacking team.
    """
    row = frames_df[
        (frames_df["match_id"] == match_id) &
        (frames_df["period"] == period) &
        (frames_df["frameIdx"] == frameIdx)
    ]
    if row.empty:
        raise ValueError(f"No frame found for match {match_id}, period {period}, frame {frameIdx}")
    
    return row.iloc[0]["lastTouch_team"]

def pitch_control_reward_fn(model, run_df, players_df, frames_df, events_df=None):
    """
    Computes pitch control reward:
        PC_with_run - PC_runner_still
    """
    end_row = run_df.iloc[-1]
    match_id = end_row["match_id"]
    period = end_row["period"]
    end_frame = end_row["frameIdx"]
    player_id = end_row["playerId"]

    attacking_side = get_attacking_side(frames_df, match_id, period, end_frame)

    runner_idx = find_runner_idx_from_players_df(
        players_df,
        match_id,
        period,
        end_frame,
        player_id
    )
        
    # Extract proper state dict for this frame:
    end_state_row = model._extract_frame_state(
        match_id,
        period,
        end_frame,
        players_df,
        frames_df
    )

    # Add frame info to state_row dict so pitch control can access it
    end_state_row["match_id"] = match_id
    end_state_row["period"] = period
    end_state_row["frameIdx"] = end_frame


    pc_with_run = compute_pitch_control(
        end_state_row,
        players_df,
        attacking_side,
        match_id,
        period,
        end_frame
    )

    pc_static = compute_pc_with_runner_still(
        end_state_row,
        runner_idx,
        players_df,
        attacking_side
    )

    delta_pc = pc_with_run - pc_static
    return delta_pc

    # --------------------------------------------------------------------------------
    # OBV REWARD FUNCTION
    # --------------------------------------------------------------------------------

def obv_reward_fn(run_df, events_df, lookahead_seconds=30.0):
    """
    Computes OBV delta reward for a run.
    """
    if events_df is None:
        return 0.0
    
    match_id = run_df["match_id"].iloc[0]
    period = run_df["period"].iloc[0]

    run_start_time = run_df["seconds_period"].min()
    run_end_time = run_df["seconds_period"].max()

    events_period = events_df[events_df["period"] == period]

    # Last event before run
    before_events = events_period[
        events_period["seconds_period"] < run_start_time
    ]
    if not before_events.empty:
        obv_before = before_events.iloc[-1]["obv_total_net"]
    else:
        obv_before = 0.0

    # First event after run
    after_events = events_period[
        (events_period["seconds_period"] > run_end_time) &
        (events_period["seconds_period"] <= run_end_time + lookahead_seconds)
    ]
    if not after_events.empty:
        obv_after = after_events.iloc[0]["obv_total_net"]
    else:
        obv_after = obv_before
    
    delta_obv = obv_after - obv_before

    if delta_obv is None or pd.isna(delta_obv):
        delta_obv = 0.0
    
    return float(delta_obv)




