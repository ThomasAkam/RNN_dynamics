from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class SimulationConfig:
    num_sequences: int = 240
    seq_len: int = 120
    obs_dim: int = 12
    process_noise_std: float = 0.02
    obs_noise_std: float = 0.08
    seed: int = 7


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_latent_nonlinear_oscillator(
    num_sequences: int,
    seq_len: int,
    process_noise_std: float,
    dt: float = 0.05,
    omega: float = 1.3,
    mu: float = 1.2,
) -> np.ndarray:
    """
    2D noisy nonlinear oscillator (stochastic Van der Pol-like system):
        x' = v
        v' = mu * (1 - x^2) * v - omega^2 * x
    Integrated with Euler-Maruyama.
    """
    z = np.zeros((num_sequences, seq_len, 2), dtype=np.float32)

    phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_sequences,))
    radius = np.random.uniform(0.6, 1.4, size=(num_sequences,))
    z[:, 0, 0] = (radius * np.cos(phase)).astype(np.float32)
    z[:, 0, 1] = (-radius * omega * np.sin(phase)).astype(np.float32)

    for t in range(seq_len - 1):
        x = z[:, t, 0]
        v = z[:, t, 1]

        dx = v
        dv = mu * (1.0 - x * x) * v - (omega * omega) * x

        noise = np.random.randn(num_sequences, 2).astype(np.float32) * process_noise_std
        z[:, t + 1, 0] = x + dt * dx + noise[:, 0]
        z[:, t + 1, 1] = v + dt * dv + noise[:, 1]

    return z


def project_latent_to_observations(
    z: np.ndarray,
    obs_dim: int,
    obs_noise_std: float,
) -> np.ndarray:
    C = np.random.randn(obs_dim, z.shape[-1]).astype(np.float32)
    y = z @ C.T
    y += np.random.randn(*y.shape).astype(np.float32) * obs_noise_std
    return y.astype(np.float32)

def simulate_observations(
    config: SimulationConfig,
    dt: float = 0.05,
    omega: float = 1.3,
    mu: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create observation trajectories from 2D noisy nonlinear oscillator latents.
    Latent dimension is fixed to 2 for this simulator.
    """
    set_seeds(config.seed)
    z = generate_latent_nonlinear_oscillator(
        num_sequences=config.num_sequences,
        seq_len=config.seq_len,
        process_noise_std=config.process_noise_std,
        dt=dt,
        omega=omega,
        mu=mu,
    )
    y = project_latent_to_observations(
        z=z,
        obs_dim=config.obs_dim,
        obs_noise_std=config.obs_noise_std,
    )
    return y, z


def visualize_generator_samples(
    observations: np.ndarray,
    latent: np.ndarray,
    output_path: str = "generator_visualization.png",
    num_sequences_to_plot: int = 8,
    num_obs_channels: int = 3,
) -> None:
    """
    Visualize simulated generator output:
    - Left: latent phase portrait (2D oscillator trajectories)
    - Right: sample observed channels over time
    """
    max_seq = min(num_sequences_to_plot, latent.shape[0])
    obs_channels = min(num_obs_channels, observations.shape[2])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    for i in range(max_seq):
        axes[0].plot(latent[i, :, 0], latent[i, :, 1], alpha=0.7, linewidth=1.2)
    axes[0].set_title("Latent Dynamics (Phase Portrait)")
    axes[0].set_xlabel("latent x")
    axes[0].set_ylabel("latent v")
    axes[0].grid(alpha=0.2)

    time = np.arange(observations.shape[1])
    seq_index = 0
    for c in range(obs_channels):
        axes[1].plot(
            time,
            observations[seq_index, :, c],
            label=f"obs[{c}]",
            linewidth=1.4,
        )
    axes[1].set_title("Observed Channels (Example Sequence)")
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("value")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
