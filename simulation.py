"""Simulated data generation for testing the GRU predictor model."""

import numpy as np


def simulate_observations(
    num_sequences,
    seq_len,
    obs_dim,
    dyn_noise_sd=0.03,
    obs_noise_sd=0.5,
    dt=0.05,
    omega=1.3,
    mu=1.2,
):
    """
    Simulate nonlinear latent trajectories and their noisy linear observations.

    The latent state is 2D and follows a noisy Van der Pol-like oscillator:
    x' = v,
    v' = mu * (1 - x^2) * v - omega^2 * x.

    Args:
        num_sequences: Number of independent trajectories to simulate.
        seq_len: Number of time steps per trajectory.
        obs_dim: Observation dimension.
        dyn_noise_sd: Standard deviation of noise in latent dynamics.
        obs_noise_sd: Standard deviation of observation noise.
        dt: Euler-Maruyama integration step size.
        omega: Oscillator frequency/stiffness parameter.
        mu: Damping parameter.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - y: Observations with shape (num_sequences, seq_len, obs_dim).
            - z: Latent trajectories with shape (num_sequences, seq_len, 2).
    """
    z = np.zeros((num_sequences, seq_len, 2), dtype=np.float32)

    z[:, 0, :] = np.random.randn(num_sequences, 2).astype(
        np.float32
    )  # Initial conditions

    # Simulate latent trajectories
    for t in range(seq_len - 1):
        x = z[:, t, 0]
        v = z[:, t, 1]

        dx = v
        dv = mu * (1.0 - x * x) * v - (omega * omega) * x

        noise = np.random.randn(num_sequences, 2).astype(np.float32) * dyn_noise_sd
        z[:, t + 1, 0] = x + dt * dx + noise[:, 0]
        z[:, t + 1, 1] = v + dt * dv + noise[:, 1]

    # Generate observations as a noisy linear projection of the latent state
    C = np.random.randn(obs_dim, z.shape[-1]).astype(np.float32)
    y = z @ C.T
    y += np.random.randn(*y.shape).astype(np.float32) * obs_noise_sd
    y = y.astype(np.float32)

    return y, z
