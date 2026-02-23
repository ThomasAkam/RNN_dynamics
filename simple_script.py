import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 

from simulation import SimulationConfig, simulate_observations
from model import GRUPredictor

# Reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Cuda config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plot_validation_results ----------------------------------------------------

def plot_validation_results(true_obs, pred_obs, gru_activity, true_latents):
    """Plot true vs predicted observations, GRU activations, and phase portraits
    for true latents and GRU activations."""

    # Setup axes.
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2)
    obs_ax = fig.add_subplot(gs[0, :])
    act_ax = fig.add_subplot(gs[1, :], sharex=obs_ax)
    true_phase_ax = fig.add_subplot(gs[2, 0])
    gru_phase_ax = fig.add_subplot(gs[2, 1])

    # Plot true vs predicted observations.
    obs_dim = true_obs.shape[2]

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, obs_dim))
    for d in range(obs_dim):
        series_color = colors[d]
        obs_ax.plot(
            true_obs[0, 1:, d].cpu().numpy(),
            color=series_color,
            linestyle="-",
            label=f"dim_{d} true",
        )
        obs_ax.plot(
            pred_obs[0, :, d].cpu().numpy(),
            color=series_color,
            linestyle="--",
            label=f"dim_{d} pred",
        )

    # Plot GRU activations.
    for unit in range(gru_activity.shape[2]):
        act_ax.plot(
            gru_activity[0, :, unit].cpu().numpy(),
            label=f"unit_{unit}",
        )

    obs_ax.legend()
    obs_ax.set_title("Predictions vs True for a Validation Sequence")
    obs_ax.set_ylabel("Observation Value")

    act_ax.legend()
    act_ax.set_title("GRU Unit Activation Timeseries")
    act_ax.set_xlabel("Time Step")
    act_ax.set_ylabel("Activation")

    # Plot phase portraits for true latents and GRU activations.
    true_phase_ax.plot(
        true_latents[0, :, 0].cpu().numpy(),
        true_latents[0, :, 1].cpu().numpy(),
    )
    true_phase_ax.set_title("True Latent Phase Portrait")
    true_phase_ax.set_xlabel("latent x")
    true_phase_ax.set_ylabel("latent v")

    gru_phase_ax.plot(
        gru_activity[0, :, 0].cpu().numpy(),
        gru_activity[0, :, 1].cpu().numpy(),
    )
    gru_phase_ax.set_title("GRU Latent Phase Portrait")
    gru_phase_ax.set_xlabel("gru unit 0")
    gru_phase_ax.set_ylabel("gru unit 1")

    fig.tight_layout()
    plt.show()

# Training
def train_example():
    # dims and hyperparams
    obs_dim = 5          # observed vector size
    gru_hidden = 2
    seq_len = 200
    n_train = 1024
    n_val = 256
    batch_size = 64
    epochs = 30
    lr = 1e-3

    # make synthetic data
    sim_cfg = SimulationConfig(
        num_sequences=n_train + n_val,
        seq_len=seq_len,
        obs_dim=obs_dim,
        process_noise_std=0.02,
        obs_noise_std=0.2,
        seed=seed,
    )
    obs, latents = simulate_observations(sim_cfg)
    train_obs = torch.from_numpy(obs[:n_train]).to(device)
    val_obs = torch.from_numpy(obs[n_train:]).to(device)
    val_latent = torch.from_numpy(latents[n_train:]).to(device) 

    model = GRUPredictor(obs_dim, gru_hidden, embed_dim=8, decode_dim=8).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    prev_val_loss = None
    val_preds = None
    gru_activity = None

    def iterate_batches(x_tensor):
        n = x_tensor.shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)
        for i in range(0, n, batch_size):
            batch_idx = indices[i : i + batch_size]
            yield x_tensor[batch_idx]

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in iterate_batches(train_obs):
            opt.zero_grad()
            preds = model(batch)                   # (B, T-1, obs_dim)
            target = batch[:, 1:, :]               # true x_{t+1}
            loss = loss_fn(preds, target)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            batches += 1
        train_loss /= batches

        # validation
        model.eval()
        with torch.no_grad():
            val_preds, gru_activity = model(val_obs, return_activations=True)
            val_loss = loss_fn(val_preds, val_obs[:, 1:, :]).item()

        if prev_val_loss is not None and val_loss > prev_val_loss:
            print(
                f"Early stopping at epoch {ep:3d}: "
                f"val_loss rose from {prev_val_loss:.6f} to {val_loss:.6f}"
            )
            break

        prev_val_loss = val_loss

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    # Plot predictions vs true and GRU activations for a validation sequence

    plot_validation_results(val_obs, val_preds, gru_activity, val_latent)

if __name__ == "__main__":
    train_example()