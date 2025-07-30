
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal
import torch.distributions as D
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from RL_reward_funcs import (compute_OBSO, c_obso_reward_fn)


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_space_size, hidden_size=256):
        super().__init__()
        input_dim = state_dim + action_space_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state_action_batch):
        return self.net(state_action_batch).squeeze(-1)
    
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_space_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space_size)
        )
    
    def forward(self, state_batch):
        return self.net(state_batch)
    
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_space_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space_size)
        )
    
    def forward(self, state_batch):
        logits = self.net(state_batch)
        return torch.softmax(logits, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state_batch):
        return self.net(state_batch).squeeze(-1)
    
class GVRNN(nn.Module):
    def __init__(
        self,
        state_dim,
        action_space_size,
        latent_dim=16,
        hidden_dim=128,
        future_steps=20
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps

        # Action embedding
        self.action_emb = nn.Embedding(action_space_size, hidden_dim)

        # Encoder: q(z_t | x_t, h_t-1)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

        # Prior: p(z_t | h_t-1)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

        # Decoder: p(x_t | z_t, h_t-1)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # predict future state
        )

        # RNN to track history
        self.rnn = nn.GRU(state_dim + latent_dim, hidden_dim)

    def forward(self, past_seq, action_idx):
        """
        Forward pass for training.

        past_seq: (T_past, batch, state_dim)
        action_idx: (batch,) int cluster indices
        """

        batch_size = past_seq.shape[1]
        action_emb = self.action_emb(action_idx)   # (batch, hidden_dim)

        h = action_emb.unsqueeze(0)  # (1, batch, hidden_dim)

        zs = []
        mus_post = []
        logvars_post = []
        recons = []

        for t in range(past_seq.shape[0]):
            x_t = past_seq[t]

            # Prior
            prior_out = self.prior(h.squeeze(0))
            mu_prior, logvar_prior = torch.chunk(prior_out, 2, dim=-1)
            logvar_prior = torch.clamp(logvar_prior, min=-10.0, max=10.0)

            # Encoder q(z|x,h)
            encoder_in = torch.cat([x_t, h.squeeze(0)], dim=-1)
            enc_out = self.encoder(encoder_in)
            mu_post, logvar_post = torch.chunk(enc_out, 2, dim=-1)
            logvar_post = torch.clamp(logvar_post, min=-10.0, max=10.0)

            z_t = self.reparameterize(mu_post, logvar_post)
            zs.append(z_t)
            mus_post.append(mu_post)
            logvars_post.append(logvar_post)

            # Decoder
            decoder_in = torch.cat([z_t, h.squeeze(0)], dim=-1)
            x_hat = self.decoder(decoder_in)
            recons.append(x_hat)

            # RNN step
            rnn_in = torch.cat([x_t, z_t], dim=-1).unsqueeze(0)
            _, h = self.rnn(rnn_in, h)

        return recons, mus_post, logvars_post

    def sample(self, past_seq, action_idx, num_samples=10):
        """
        Generate samples of future sequences.
        """
        batch_size = past_seq.shape[1]
        action_emb = self.action_emb(action_idx)
        h = action_emb.unsqueeze(0)

        samples = []

        for n in range(num_samples):
            traj = []
            h_t = h

            for t in range(self.future_steps):
                # Prior p(z | h)
                prior_out = self.prior(h_t.squeeze(0))
                mu_prior, logvar_prior = torch.chunk(prior_out, 2, dim=-1)
                logvar_prior = torch.clamp(logvar_prior, min=-10.0, max=10.0)

                z_t = self.reparameterize(mu_prior, logvar_prior)

                decoder_in = torch.cat([z_t, h_t.squeeze(0)], dim=-1)
                x_hat = self.decoder(decoder_in)

                traj.append(x_hat)

                # RNN step
                rnn_in = torch.cat([x_hat, z_t], dim=-1).unsqueeze(0)
                _, h_t = self.rnn(rnn_in, h_t)

            traj = torch.stack(traj, dim=0)
            samples.append(traj)

        return samples

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

class TrackingDataset(Dataset):
    """
    Dataset of:
       past_seq: (T_past, state_dim)
       future_seq: (T_future, state_dim)
       action_idx: int
    """
    def __init__(self, data_tuples):
        self.data = data_tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# KL Divergence Function
    
def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL between N(mu_q, sigma_q) || N(mu_p, sigma_p)
    """
    return 0.5 * torch.sum(
        logvar_p - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        - 1,
        dim=-1
    )
    
# -----------------------
# Training function
# -----------------------

def train_gvrnn(
    gvrnn_model,
    dataset,
    num_epochs=50,
    batch_size=64,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    gvrnn_model = gvrnn_model.to(device)
    optimizer = torch.optim.Adam(gvrnn_model.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_rec_loss = 0.0
        total_kl = 0.0

        for past_seq, future_seq, action_idx in loader:
            past_seq = past_seq.to(device)         # (T_past, B, state_dim)
            future_seq = future_seq.to(device)     # (T_future, B, state_dim)
            action_idx = action_idx.to(device)     # (B,)

            past_seq = past_seq.transpose(0, 1)    # → (T_past, B, state_dim)
            future_seq = future_seq.transpose(0, 1)

            # Forward pass over past seq
            recons, mus_post, logvars_post = gvrnn_model(past_seq, action_idx)

            # Prior terms
            h = gvrnn_model.action_emb(action_idx).unsqueeze(0)  # (1, B, H)
            kl_total = 0.0

            rec_loss = 0.0
            for t, (x_hat, mu_post, logvar_post) in enumerate(zip(recons, mus_post, logvars_post)):
                # Ground truth for past
                x_true = past_seq[t]

                # Prior for timestep t
                prior_out = gvrnn_model.prior(h.squeeze(0))
                mu_prior, logvar_prior = torch.chunk(prior_out, 2, dim=-1)

                # KL divergence
                kl = kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior)
                kl_total += kl.mean()

                # Reconstruction loss
                rec_loss += torch.mean((x_hat - x_true) ** 2)

                # Update RNN hidden state
                z_t = gvrnn_model.reparameterize(mu_post, logvar_post)
                rnn_in = torch.cat([x_true, z_t], dim=-1).unsqueeze(0)
                _, h = gvrnn_model.rnn(rnn_in, h)

            # Mean over timesteps
            rec_loss = rec_loss / len(recons)

            loss = rec_loss + kl_total * 0.1  # weight KL if desired

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_kl += kl_total.item()

        avg_loss = total_loss / len(loader)
        avg_rec = total_rec_loss / len(loader)
        avg_kl = total_kl / len(loader)
        print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Recon={avg_rec:.4f}, KL={avg_kl:.4f}")

def build_tracking_dataset(
    runs_df,
    assignments_df,
    players_df,
    frames_df,
    model,
    T_past=10,
    T_future=10
):
    dataset = []

    for run_id, run_df in tqdm(runs_df.groupby("run_id"), desc="Building tracking dataset", total=runs_df["run_id"].nunique()):
        if len(run_df) < 4:
            continue

        match_id = run_df["match_id"].iloc[0]
        period = run_df["period"].iloc[0]
        action = run_df["assigned_cluster"].iloc[0]

        # split into past and future
        split_idx = int(len(run_df) * 0.5)
        if split_idx < 1:
            continue

        past_df = run_df.iloc[:split_idx]
        future_df = run_df.iloc[split_idx:]

        if past_df.empty or future_df.empty:
            continue

        # get state rows for each frame
        past_states = []
        for f in past_df["frameIdx"]:
            state_row = model._extract_frame_state(match_id, period, f, players_df, frames_df)
            if state_row is None:
                continue
            past_states.append(model.state_fn(state_row))
        if len(past_states) == 0:
            continue

        # pad/truncate
        past_states = np.stack(past_states, axis=0)
        if past_states.shape[0] < T_past:
            pad_len = T_past - past_states.shape[0]
            pad = np.zeros((pad_len, past_states.shape[1]))
            past_states = np.concatenate([past_states, pad], axis=0)
        else:
            past_states = past_states[:T_past]

        future_states = []
        for f in future_df["frameIdx"]:
            state_row = model._extract_frame_state(match_id, period, f, players_df, frames_df)
            if state_row is None:
                continue
            future_states.append(model.state_fn(state_row))
        if len(future_states) == 0:
            continue

        future_states = np.stack(future_states, axis=0)
        if future_states.shape[0] < T_future:
            pad_len = T_future - future_states.shape[0]
            pad = np.zeros((pad_len, future_states.shape[1]))
            future_states = np.concatenate([future_states, pad], axis=0)
        else:
            future_states = future_states[:T_future]

        dataset.append((
            torch.tensor(past_states, dtype=torch.float32),
            torch.tensor(future_states, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long)
        ))

    print(f"Built GVRNN dataset with {len(dataset)} sequences.")
    return TrackingDataset(dataset)


class FootballOffBallRL(BaseEstimator):
    """
    RL pipeline for off-ball runs in football.
    
    This class:
    - stores state/action/reward/next-state data
    - fits an RL model (e.g. fitted Q iteration)
    - ranks players by long-term run value
    
    Customisable:
    - State feature extraction
    - Reward functions
    - Learning algorithms
    """

    def __init__(
        self,
        method="lightgbm",        # "lightgbm" or "dqn" or "actor-critic"
        gamma=0.9,
        n_iters=10,
        action_space_size=70,
        regressor=None,           # Used only for LightGBM
        state_fn=None,
        reward_fn=None,
        # DQN specific:
        state_dim=None,
        hidden_size=256,
        batch_size=128,
        learning_rate=1e-3,
        device="cpu",
        target_update_freq=1000,  # How often to update target network
        reward_func="obv",
        gvrnn_model=None,         # Optional GVRNN model for future prediction
    ):
        self.method = method
        self.gamma = gamma
        self.n_iters = n_iters
        self.action_space_size = action_space_size
        self.state_fn = state_fn
        self.reward_fn = reward_fn
        self.target_update_freq = target_update_freq
        self.reward_func = reward_func
        self.gvrnn_model = gvrnn_model

        if method == "lightgbm":
            self.regressor = regressor if regressor is not None else LGBMRegressor()
        elif method == "dqn":
            assert state_dim is not None, "state_dim required for DQN"
            self.q_network = QNetwork(
                state_dim=state_dim,
                action_space_size=action_space_size,
                hidden_size=hidden_size
            ).to(device)

            # NEW: Target network
            self.target_q_network = QNetwork(
                state_dim=state_dim,
                action_space_size=action_space_size,
                hidden_size=hidden_size
            ).to(device)

            # Initially identical
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_q_network.eval()

            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()
            self.device = device
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq or 1000

        elif method == "actor_critic":
            assert state_dim is not None, "state_dim required for Actor-Critic"
            
            self.actor = ActorNetwork(
                state_dim=state_dim,
                action_space_size=action_space_size,
                hidden_size=hidden_size
            ).to(device)

            self.critic = CriticNetwork(
                state_dim=state_dim,
                hidden_size=hidden_size
            ).to(device)

            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()
            self.device = device
            self.batch_size = batch_size
        else:
            raise ValueError("method must be either 'lightgbm', 'dqn', or 'actor_critic'")
       
        
    def update_target_network(self):
        """
        Copies weights from the online network to the target network.
        """
        self.target_q_network.load_state_dict(self.q_network.state_dict())
                
    def prepare_data(self, runs_df, players_df, frames_df, assignments_df, events_dfs_by_match):
        """
        Constructs RL dataset:
            run_id → (state, action, reward, next_state, player_id)
        """
        print("Preparing RL dataset...")
        data_rows = []
        
        grouped = runs_df.groupby("run_id")
        
        for run_id, run_df in tqdm(grouped, total=len(grouped)):
            match_id = run_df["match_id"].iloc[0]
            period = run_df["period"].iloc[0]
            player_id = run_df["playerId"].iloc[0]
            start_frame = run_df["frameIdx"].min()
            end_frame = run_df["frameIdx"].max()

            # ---------------------------------------------
            # 1. State at start
            # ---------------------------------------------
            start_state_row = self._extract_frame_state(
                match_id, period, start_frame, players_df, frames_df
            )
            if start_state_row is None:
                continue

            state_vector = self.state_fn(start_state_row)
            
            # ---------------------------------------------
            # 2. Next State after run
            # ---------------------------------------------
            end_state_row = self._extract_frame_state(
                match_id, period, end_frame, players_df, frames_df
            )
            if end_state_row is None:
                continue

            next_state_vector = self.state_fn(end_state_row)

            # ---------------------------------------------
            # 3. Action
            # ---------------------------------------------
            assigned_cluster = assignments_df.loc[
                assignments_df["run_id"] == run_id, "assigned_cluster"
            ]
            if assigned_cluster.empty:
                continue
            action = int(assigned_cluster.values[0])
            
            # ---------------------------------------------
            # 4. Reward
            # ---------------------------------------------
            # if self.reward_fn is not None:
            #     # Extract events for this match
            #     events_df = events_dfs_by_match.get(match_id, None)
            #     if events_df is None:
            #         reward = 0.0
            #     else:
            #         reward = self.reward_fn(run_df, events_df)
            #         if reward is None:
            #             reward = 0.0
            # else:
            #     reward = 0.0
            if self.reward_fn is not None:
                match_key = match_id.replace(".json", "")
                events_df = events_dfs_by_match.get(match_key, None)
                if events_df is None:
                    print(f"[WARN] No events found for match_id: {match_key}")

                # earlier in prepare_data:
                end_state_row = self._extract_frame_state(
                    match_id, period, end_frame, players_df, frames_df
                )
                if end_state_row is None:
                    continue

                next_state_vector = self.state_fn(end_state_row)

                # attach frame info for reward calc
                end_state_row.update({
                    "match_id": match_id,
                    "period": period,
                    "frameIdx": end_frame
                })
                # reward
                if self.reward_func == "c-obso":
                    reward = self.reward_fn(run_df, players_df, frames_df)
                elif self.reward_func == "pc":
                    # Pitch control reward needs extra arguments
                    reward = self.reward_fn(
                        self,
                        run_df,
                        players_df,
                        frames_df,
                        events_df
                    ) if events_df is not None else 0.0
                else:
                    # simpler reward function
                    reward = self.reward_fn(
                        run_df,
                        events_df
                    ) if events_df is not None else 0.0

            
            data_rows.append({
                "run_id": run_id,
                "state": state_vector,
                "action": action,
                "reward": reward,
                "next_state": next_state_vector,
                "playerId": player_id
            })

        self.rl_data_ = pd.DataFrame(data_rows)
        print(f"Prepared {len(self.rl_data_)} RL transitions.")
        return self.rl_data_
    
    def _extract_frame_state(self, match_id, period, frame_idx, players_df, frames_df):
        """
        Helper to extract a single frame's state.
        """

        if self.reward_func in ("pc", "c-obso"):  
            players_at_frame = players_df[
            (players_df["match_id"] == match_id) &
            (players_df["period"] == period) &
            (players_df["frameIdx"] == frame_idx)]
            if len(players_at_frame) < 22:
                return None

            player_coords = []
            player_velocities = []
            for _, p in players_at_frame.iterrows():
                player_coords.extend([p["x"], p["y"]])
                player_velocities.extend([p["vx"], p["vy"]])

            ball_at_frame = frames_df[
                (frames_df["match_id"] == match_id) &
                (frames_df["period"] == period) &
                (frames_df["frameIdx"] == frame_idx)
            ]
            if ball_at_frame.empty:
                return None

            ball_x = ball_at_frame["ball_x"].iloc[0]
            ball_y = ball_at_frame["ball_y"].iloc[0]

            return {
                "player_coords": np.array(player_coords),
                "player_velocities": np.array(player_velocities),
                "ball_coords": np.array([ball_x, ball_y])
            }
        
        else: 
    
            players_at_frame = players_df[
                (players_df["match_id"] == match_id) &
                (players_df["period"] == period) &
                (players_df["frameIdx"] == frame_idx)
            ]
            if len(players_at_frame) < 22:
                return None
            
            player_coords = []
            for _, p in players_at_frame.iterrows():
                player_coords.extend([p["x"], p["y"]])

            ball_at_frame = frames_df[
                (frames_df["match_id"] == match_id) &
                (frames_df["period"] == period) &
                (frames_df["frameIdx"] == frame_idx)
            ]
            if ball_at_frame.empty:
                return None
            
            ball_x = ball_at_frame["ball_x"].iloc[0]
            ball_y = ball_at_frame["ball_y"].iloc[0]

            return {
                "player_coords": np.array(player_coords),
                "ball_coords": np.array([ball_x, ball_y])
            }
        
    def compute_counterfactual_value(self, run_df, players_df, frames_df, K=20):
        """
        Compare run's observed C-OBSO to counterfactual GVRNN samples.
        """
        if self.gvrnn_model is None:
            raise ValueError("GVRNN model is not loaded.")

        # --- 1. Past context
        match_id = run_df["match_id"].iloc[0]
        period = run_df["period"].iloc[0]
        start_frame = run_df["frameIdx"].min()

        start_state_row = self._extract_frame_state(
            match_id, period, start_frame, players_df, frames_df
        )
        state_vector = self.state_fn(start_state_row)
        past_seq = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(1).unsqueeze(0)

        # --- 2. Action
        action = run_df["assigned_cluster"].iloc[0]
        action_idx = torch.tensor([action])

        # --- 3. Sample futures
        sampled_trajs = self.gvrnn_model.sample(past_seq, action_idx, num_samples=K)

        # Compute C-OBSO for each trajectory
        sampled_obsos = []
        for traj in sampled_trajs:
            last_pos = traj[-1, 0].detach().cpu().numpy()[:2]  # assuming [x,y,...]
            end_state_row = {
                "match_id": match_id,
                "period": period,
                "frameIdx": start_frame,
                "ball_coords": last_pos
            }
            obso_val = compute_OBSO(end_state_row, players_df, frames_df)
            sampled_obsos.append(obso_val)

        # observed run's C-OBSO
        obs_value_real = c_obso_reward_fn(run_df, players_df, frames_df)

        mean_sample = np.mean(sampled_obsos)
        std_sample = np.std(sampled_obsos)
        z_score = (obs_value_real - mean_sample) / (std_sample + 1e-8)

        return {
            "observed_value": obs_value_real,
            "mean_counterfactual": mean_sample,
            "std_counterfactual": std_sample,
            "z_score": z_score
        }
        
    def fit(self):
        if self.method == "lightgbm":
            self._fit_lightgbm()
        elif self.method == "dqn":
            self._fit_dqn()
        elif self.method == "actor_critic":
            self._fit_actor_critic()
        else:
            raise ValueError("Invalid method")
        
    # --------------------
    # LightGBM FQI
    # --------------------

    def _fit_lightgbm(self):
        df = self.rl_data_
        X = self._build_features(df["state"], df["action"])
        y = df["reward"].values.copy()

        for it in range(self.n_iters):
            self.regressor.fit(X, y)
            next_Q_vals = self.predict_next_state_values(df["next_state"])
            y = df["reward"].values + self.gamma * next_Q_vals
            print(f"[LightGBM] Iteration {it+1}/{self.n_iters} done.")
        
        self.fitted_ = True

    # --------------------
    # DQN
    # --------------------

    def _fit_dqn(self):
        df = self.rl_data_

        states = np.stack(df["state"].values)
        actions = df["action"].values
        rewards = df["reward"].values
        next_states = np.stack(df["next_state"].values)

        dataset_size = len(df)

        for epoch in range(self.n_iters):
            perm = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start+self.batch_size]
                
                s_batch = torch.tensor(states[idx], dtype=torch.float32, device=self.device)
                a_batch = torch.tensor(actions[idx], dtype=torch.int64, device=self.device)
                r_batch = torch.tensor(rewards[idx], dtype=torch.float32, device=self.device)
                s2_batch = torch.tensor(next_states[idx], dtype=torch.float32, device=self.device)

                # Create (s,a)
                onehot_a = torch.eye(self.action_space_size, device=self.device)[a_batch]
                sa_batch = torch.cat([s_batch, onehot_a], dim=1)

                # # Compute target:
                # with torch.no_grad():
                #     target_q_vals = self.target_q_network(s2_batch)   # shape (batch, n_actions)
                #     max_q_next = target_q_vals.max(dim=1).values

                #     y_batch = r_batch + self.gamma * max_q_next

                            # Double Q-Learning target computation
                with torch.no_grad():
                    online_next_qs = self.q_network(s2_batch)        # (batch, n_actions)
                    best_next_actions = online_next_qs.argmax(dim=1) # (batch,)

                    target_qs = self.target_q_network(s2_batch)      # (batch, n_actions)
                    max_q_next = target_qs.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

                    y_batch = r_batch + self.gamma * max_q_next

                # Compute Q(s,a) for taken actions:
                online_q_vals = self.q_network(s_batch)
                q_pred = online_q_vals.gather(1, a_batch.unsqueeze(1)).squeeze(1)

                # Loss and update
                loss = self.loss_fn(q_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"[DQN] Epoch {epoch+1}/{self.n_iters} done.")
        
        self.fitted_ = True

    def _fit_actor_critic(self):
        df = self.rl_data_

        states = np.stack(df["state"].values)
        actions = df["action"].values
        rewards = df["reward"].values
        next_states = np.stack(df["next_state"].values)

        dataset_size = len(df)

        for epoch in range(self.n_iters):
            perm = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start+self.batch_size]

                s_batch = torch.tensor(states[idx], dtype=torch.float32, device=self.device)
                a_batch = torch.tensor(actions[idx], dtype=torch.int64, device=self.device)
                r_batch = torch.tensor(rewards[idx], dtype=torch.float32, device=self.device)
                s2_batch = torch.tensor(next_states[idx], dtype=torch.float32, device=self.device)

                # Critic target
                with torch.no_grad():
                    v_next = self.critic(s2_batch)
                    target_v = r_batch + self.gamma * v_next

                v = self.critic(s_batch)
                critic_loss = self.loss_fn(v, target_v)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

                # Actor update
                probs = self.actor(s_batch)  # (batch, n_actions)
                eps = 1e-8
                log_probs = torch.log(torch.clamp(probs.gather(1, a_batch.unsqueeze(1)).squeeze(1), min = eps))
                advantage = (target_v - v).detach()
                actor_loss = -(log_probs * advantage).mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

            print(f"[ActorCritic] Epoch {epoch+1}/{self.n_iters} done.")
        
        self.fitted_ = True

    
    # --------------------
    # Q prediction
    # --------------------

    def predict_next_state_values(self, next_states):
        if self.method == "lightgbm":
            all_qs = []
            for a in range(self.action_space_size):
                feats = self._build_features(next_states, [a] * len(next_states))
                qs = self.regressor.predict(feats)
                all_qs.append(qs)
            all_qs = np.stack(all_qs, axis=1)
            return np.max(all_qs, axis=1)
        elif self.method == "dqn":
            next_states_tensor = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
            all_qs = []
            next_qs = self.q_network(next_states_tensor)   # (batch, n_actions)
            return next_qs.max(dim=1).values.cpu().numpy()
    

    def Q(self, state, action):
        if self.method == "lightgbm":
            X = self._build_features([state], [action])
            return self.regressor.predict(X)[0]
        elif self.method == "dqn":
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)   # shape (1, n_actions)
                return q_values[0, action].item()
            
        elif self.method == "actor_critic":
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                v_value = self.critic(state_tensor).item()
                probs = self.actor(state_tensor).squeeze(0)
                action_prob = probs[action].item()
            # For ranking purposes, we can multiply:
            return action_prob * v_value
    
    def _build_features(self, states, actions):
        """
        Build combined feature vector for regressor:
            state + one-hot(action)
        """
        states_arr = np.stack(states)
        actions_onehot = np.eye(self.action_space_size)[actions]
        return np.concatenate([states_arr, actions_onehot], axis=1)
    
    def rank_players(self):
        """
        Aggregate Q-values for all players.
        """
        df = self.rl_data_.copy()
        df["q_value"] = df.apply(
            lambda row: self.Q(row["state"], row["action"]),
            axis=1
        )
        player_values = df.groupby("playerId")["q_value"].mean()
        return player_values.sort_values(ascending=False)
    
    def plot_player_q_distribution(self, player_id=None, bins=50):
        """
        Plots the histogram of Q-values for:
            - one specific player (if player_id given), OR
            - all players combined
        
        Parameters
        ----------
        player_id : str or None
            Player UUID to plot. If None, plots for all players combined.
        bins : int
            Number of histogram bins.
        """
        df = self.rl_data_.copy()

        # Compute Q-values if not yet stored
        if "q_value" not in df.columns:
            df["q_value"] = df.apply(
                lambda row: self.Q(row["state"], row["action"]),
                axis=1
            )

        if player_id is not None:
            df = df[df["playerId"] == player_id]
            if df.empty:
                print(f"No data found for playerId: {player_id}")
                return

            title = f"Q-value distribution for player {player_id}"
        else:
            title = "Q-value distribution across all players"

        plt.figure(figsize=(8, 5))
        sns.histplot(df["q_value"], bins=bins, kde=True, color="steelblue")
        plt.title(title)
        plt.xlabel("Q-value")
        plt.ylabel("Frequency")
        plt.show()