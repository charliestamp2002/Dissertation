import numpy as np

def simple_state_fn(state_row):
    """
    Converts state_row into a flat vector:
    [player_x1, player_y1, ..., player_x22, player_y22, ball_x, ball_y]
    """
    player_coords = state_row["player_coords"]
    ball_coords = state_row["ball_coords"]
    return np.concatenate([player_coords, ball_coords])

def c_obso_state_fn(state_row):
    """
    Constructs the state vector for C-OBSO.
    """
    return np.concatenate([
        state_row["player_coords"],        # (44,)
        state_row["player_velocities"],    # (44,)
        state_row["ball_coords"]           # (2,)
    ])
