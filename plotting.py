import matplotlib.pyplot as plt
import numpy as np


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
