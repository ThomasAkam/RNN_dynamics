from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from model import TinyGRUPredictor
from simulation import (
    SimulationConfig,
    simulate_observations,
    visualize_generator_samples,
)


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 120
    k_folds: int = 5
    pre_mlp_dim: int | None = None
    seed: int = 7


def kfold_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    splits = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def one_step_r2(y_true, y_pred):
    """
    R^2 over all batches/timesteps/dimensions for one-step-ahead predictions.
    """
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    sse = torch.sum((y_true_flat - y_pred_flat) ** 2)
    sst = torch.sum((y_true_flat - torch.mean(y_true_flat)) ** 2)
    r2 = 1.0 - (sse / (sst + 1e-8))
    return float(r2.item())


def train_one_fold(
    data_train,
    data_val,
    obs_dim,
    hidden_units,
    config,
    device,
):
    model = TinyGRUPredictor(
        obs_dim=obs_dim,
        hidden_units=hidden_units,
        pre_mlp_dim=config.pre_mlp_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(data_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    for _ in range(config.epochs):
        for (batch_obs,) in train_loader:
            batch_obs = batch_obs.to(device)
            pred_all, _ = model(batch_obs)
            pred_next = pred_all[:, :-1, :]
            target_next = batch_obs[:, 1:, :]

            loss = loss_fn(pred_next, target_next)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_obs = data_val.to(device)
        val_pred_all, val_hidden = model(val_obs)
        val_pred_next = val_pred_all[:, :-1, :]
        val_target_next = val_obs[:, 1:, :]
        val_mse = float(loss_fn(val_pred_next, val_target_next).item())
        val_r2 = one_step_r2(val_target_next, val_pred_next)

    return {
        "val_mse": val_mse,
        "val_r2": val_r2,
        "hidden_norm": float(val_hidden.norm(dim=-1).mean().item()),
    }


def cross_validated_experiment(
    observations,
    hidden_unit_list,
    config,
):
    n_seq, _, obs_dim = observations.shape
    splits = kfold_indices(n=n_seq, k=config.k_folds, seed=config.seed)
    data_t = torch.tensor(observations, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for hidden_units in tqdm(hidden_unit_list, desc="Hidden units"):
        fold_mse = []
        fold_r2 = []
        fold_hidden_norm = []

        for train_idx, val_idx in splits:
            fold_result = train_one_fold(
                data_train=data_t[train_idx],
                data_val=data_t[val_idx],
                obs_dim=obs_dim,
                hidden_units=hidden_units,
                config=config,
                device=device,
            )
            fold_mse.append(fold_result["val_mse"])
            fold_r2.append(fold_result["val_r2"])
            fold_hidden_norm.append(fold_result["hidden_norm"])

        results[hidden_units] = {
            "mse_mean": float(np.mean(fold_mse)),
            "mse_std": float(np.std(fold_mse)),
            "r2_mean": float(np.mean(fold_r2)),
            "r2_std": float(np.std(fold_r2)),
            "hidden_norm_mean": float(np.mean(fold_hidden_norm)),
        }

    return results


def cross_validated_pre_mlp_experiment(
    observations,
    pre_mlp_dims,
    hidden_units,
    base_config,
):
    n_seq, _, obs_dim = observations.shape
    splits = kfold_indices(n=n_seq, k=base_config.k_folds, seed=base_config.seed)
    data_t = torch.tensor(observations, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for pre_mlp_dim in tqdm(pre_mlp_dims, desc="Input MLP"):
        fold_mse = []
        fold_r2 = []
        fold_hidden_norm = []

        config = TrainConfig(
            learning_rate=base_config.learning_rate,
            weight_decay=base_config.weight_decay,
            batch_size=base_config.batch_size,
            epochs=base_config.epochs,
            k_folds=base_config.k_folds,
            pre_mlp_dim=pre_mlp_dim,
            seed=base_config.seed,
        )

        for train_idx, val_idx in splits:
            fold_result = train_one_fold(
                data_train=data_t[train_idx],
                data_val=data_t[val_idx],
                obs_dim=obs_dim,
                hidden_units=hidden_units,
                config=config,
                device=device,
            )
            fold_mse.append(fold_result["val_mse"])
            fold_r2.append(fold_result["val_r2"])
            fold_hidden_norm.append(fold_result["hidden_norm"])

        results[pre_mlp_dim] = {
            "mse_mean": float(np.mean(fold_mse)),
            "mse_std": float(np.std(fold_mse)),
            "r2_mean": float(np.mean(fold_r2)),
            "r2_std": float(np.std(fold_r2)),
            "hidden_norm_mean": float(np.mean(fold_hidden_norm)),
        }

    return results


def print_results(
    results,
    label,
):
    print(f"\nCross-validated one-step prediction accuracy vs {label}")
    print(f"{label} | mse_mean ± mse_std | r2_mean ± r2_std")
    print("-" * 58)
    def sort_key(value):
        return (value is None, value if value is not None else -1)

    for key in sorted(results.keys(), key=sort_key):
        r = results[key]
        display = "None" if key is None else str(key)
        print(
            f"{display:>5} | "
            f"{r['mse_mean']:.5f} ± {r['mse_std']:.5f} | "
            f"{r['r2_mean']:.5f} ± {r['r2_std']:.5f}"
        )


def run_experiment():
    sim_config = SimulationConfig(
        num_sequences=40,
        seq_len=50,
        obs_dim=8,
        process_noise_std=0.02,
        obs_noise_std=0.08,
        seed=7,
    )

    train_config = TrainConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        epochs=120,
        k_folds=2,
        pre_mlp_dim=None,
        seed=7,
    )

    print("Generating simulated data (2D nonlinear oscillator latents)...")
    observations, latent = simulate_observations(sim_config)
    print(
        f"Simulated {observations.shape[0]} sequences, "
        f"length={observations.shape[1]}, "
        f"obs_dim={observations.shape[2]}, latent_dim={latent.shape[2]}"
    )

    visualize_generator_samples(
        observations=observations,
        latent=latent,
        output_path="generator_visualization.png",
    )
    print("Saved generator visualization to generator_visualization.png")

    print("\nExperiment 1: Vary GRU hidden units")
    hidden_unit_list = [1, 2, 4]
    print(f"Evaluating hidden units: {hidden_unit_list}")

    results_units = cross_validated_experiment(
        observations=observations,
        hidden_unit_list=hidden_unit_list,
        config=train_config,
    )
    print_results(results_units, label="units")

    # print("\nExperiment 2: Vary input MLP size")
    # fixed_hidden_units = 2
    # pre_mlp_dims = [None, 4, 8]
    # print(f"Evaluating pre-MLP sizes: {pre_mlp_dims} (hidden_units={fixed_hidden_units})")

    # results_mlp = cross_validated_pre_mlp_experiment(
    #     observations=observations,
    #     pre_mlp_dims=pre_mlp_dims,
    #     hidden_units=fixed_hidden_units,
    #     base_config=train_config,
    # )
    # print_results(results_mlp, label="pre_mlp")
