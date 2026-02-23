import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from simulation import SimulationConfig, simulate_observations
from model import GRUPredictor
from plotting import plot_validation_results
from training import train_model

# Reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Cuda config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dims and hyperparams
obs_dim = 5          # observed vector size
gru_hidden = 2
seq_len = 200
n_train = 1024
n_val = 256
batch_size = 64
epochs = 50
lr = 1e-2
reg_lambda = 1
weight_decay = 0.01

# Generate data
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
val_latents = torch.from_numpy(latents[n_train:]).to(device) 

# Setup model, optimizer, and loss function
model = GRUPredictor(obs_dim, gru_hidden, embed_dim=8, decode_dim=8).to(device)
opt = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Train model
val_preds, val_activity = train_model(
    model=model,
    train_obs=train_obs,
    val_obs=val_obs,
    epochs=epochs,
    batch_size=batch_size,
    optimizer=opt,
    loss_fn=loss_fn,
    reg_lambda=reg_lambda,
    weight_decay=weight_decay,
)

# Plot predictions vs true and GRU activations for a validation sequence

plot_validation_results(val_obs, val_preds, val_activity, val_latents)

