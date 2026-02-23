import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
from simulation import SimulationConfig, simulate_observations

# Reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model
class GRUPredictor(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_size, batch_first=True)
        self.output_map = nn.Linear(hidden_size, obs_dim)

    def forward(self, x, return_activations=False):
        outputs, _ = self.gru(x[:, :-1])
        preds = self.output_map(outputs)
        if return_activations:
            return preds, outputs
        return preds


class GRUPredictorWithMLP(nn.Module):
    def __init__(self, obs_dim, hidden_size, embed_dim=8, decode_dim=8):
        super().__init__()
        self.input_embed = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.output_decode = nn.Sequential(
            nn.Linear(hidden_size, decode_dim),
            nn.ReLU(),
            nn.Linear(decode_dim, obs_dim),
        )

    def forward(self, x, return_activations=False):
        embedded = self.input_embed(x[:, :-1])
        outputs, _ = self.gru(embedded)
        preds = self.output_decode(outputs)
        if return_activations:
            return preds, outputs
        return preds



# Training
def train_example():
    # dims and hyperparams
    obs_dim = 5          # observed vector size
    gru_hidden = 2
    seq_len = 500
    n_train = 1024
    n_val = 256
    batch_size = 64
    epochs = 1000
    lr = 1e-3

    # make synthetic data
    train_cfg = SimulationConfig(
        num_sequences=n_train,
        seq_len=seq_len,
        obs_dim=obs_dim,
        process_noise_std=0.01,
        obs_noise_std=0.1,
        seed=seed,
    )
    val_cfg = SimulationConfig(
        num_sequences=n_val,
        seq_len=seq_len,
        obs_dim=obs_dim,
        process_noise_std=0.01,
        obs_noise_std=0.1,
        seed=seed + 1,
    )
    train_obs, _ = simulate_observations(train_cfg)
    val_obs, val_latent = simulate_observations(val_cfg)
    train_x = torch.from_numpy(train_obs).to(device)
    val_x = torch.from_numpy(val_obs).to(device)
    val_z = torch.from_numpy(val_latent).to(device)

    #model = GRUPredictor(obs_dim, gru_hidden).to(device)
    model = GRUPredictorWithMLP(obs_dim, gru_hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    prev_val_loss = None

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
        for batch in iterate_batches(train_x):
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
            preds_val = model(val_x)
            val_loss = loss_fn(preds_val, val_x[:, 1:, :]).item()

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
    seq = val_x[:1]                       # (1, T, obs_dim)
    with torch.no_grad():
        preds, activations = model(seq, return_activations=True)
    true = seq[:, 1:, :]

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2)
    obs_ax = fig.add_subplot(gs[0, :])
    act_ax = fig.add_subplot(gs[1, :], sharex=obs_ax)
    true_phase_ax = fig.add_subplot(gs[2, 0])
    gru_phase_ax = fig.add_subplot(gs[2, 1])

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, obs_dim))
    for d in range(obs_dim):
        series_color = colors[d]
        obs_ax.plot(
            true[0, :, d].cpu().numpy(),
            color=series_color,
            linestyle="-",
            label=f"dim_{d} true",
        )
        obs_ax.plot(
            preds[0, :, d].cpu().numpy(),
            color=series_color,
            linestyle="--",
            label=f"dim_{d} pred",
        )

    for unit in range(gru_hidden):
        act_ax.plot(
            activations[0, :, unit].cpu().numpy(),
            label=f"unit_{unit}",
        )

    obs_ax.legend()
    obs_ax.set_title("Predictions vs True for a Validation Sequence")
    obs_ax.set_ylabel("Observation Value")

    act_ax.legend()
    act_ax.set_title("GRU Unit Activation Timeseries")
    act_ax.set_xlabel("Time Step")
    act_ax.set_ylabel("Activation")

    true_phase_ax.plot(
        val_z[0, :, 0].cpu().numpy(),
        val_z[0, :, 1].cpu().numpy(),
    )
    true_phase_ax.set_title("True Latent Phase Portrait")
    true_phase_ax.set_xlabel("latent x")
    true_phase_ax.set_ylabel("latent v")

    gru_phase_ax.plot(
        activations[0, :, 0].cpu().numpy(),
        activations[0, :, 1].cpu().numpy(),
    )
    gru_phase_ax.set_title("GRU Latent Phase Portrait")
    gru_phase_ax.set_xlabel("gru unit 0")
    gru_phase_ax.set_ylabel("gru unit 1")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_example()