"""Main script to run the full pipeline: data generation, model training, and plotting."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from simulation import simulate_observations
from model import GRUPredictor
from plotting import plot_validation_results
from training import train_model

# Config ----------------------------------------------------------------------

# Random seeds
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Cuda configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters ------------------------------------------------------------------

# Data generation
obs_dim = 5  # observation vector size
seq_len = 300  # time steps per sequence
n_train = 1024  # number of training sequences
n_val = 256  # number of validation sequences
dyn_noise_sd = 0.02  # SD of noise in latent dynamics
obs_noise_sd = 0.5  # SD of noise in observations

# GRU model
gru_dim = 2  # Number of GRU hidden units.
embed_dim = 8  # input ReLU layer size, set to None to disable layer.
decode_dim = 8  # output ReLU layer size, set to None to disable layer.

# Training
batch_size = 64  # sequences per optimization step
epochs = 50  # max full passes through training data
lr = 1e-2  # optimizer learning rate
reg_lambda = 1  # GRU activity regularization strength
weight_decay = 0.01  # L2 weight regularization strength

# Generate data ---------------------------------------------------------------

obs, latents = simulate_observations(
    num_sequences=n_train + n_val,
    seq_len=seq_len,
    obs_dim=obs_dim,
    dyn_noise_sd=dyn_noise_sd,
    obs_noise_sd=obs_noise_sd,
)

train_obs = torch.from_numpy(obs[:n_train]).to(device)
val_obs = torch.from_numpy(obs[n_train:]).to(device)
val_latents = torch.from_numpy(latents[n_train:]).to(device)

# Setup model, optimizer, and loss function -----------------------------------

model = GRUPredictor(obs_dim, gru_dim, embed_dim, decode_dim).to(device)
opt = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Train model ---------------------------------------------------------------

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

# Plot predictions vs true and GRU activations for a validation sequence ------

plot_validation_results(
    val_obs, val_preds, val_activity, val_latents, save_path="validation_results.png"
)
