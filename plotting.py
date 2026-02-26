import matplotlib.pyplot as plt
import numpy as np


def plot_validation_results(
    true_obs, pred_obs, gru_activity, true_latents, save_path=None
):
    """Plot true vs predicted observations, GRU activations, and phase portraits
    for true latents and GRU activations."""

    # Setup axes.
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)
    obs_ax = fig.add_subplot(gs[0, :])
    act_ax = fig.add_subplot(gs[1, :], sharex=obs_ax)
    true_phase_ax = fig.add_subplot(gs[2, 0])
    gru_phase_ax = fig.add_subplot(gs[2, 1])

    # Plot true vs predicted observations.
    obs_dim = true_obs.shape[2]

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, obs_dim))
    for d in range(min(obs_dim, 3)):
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

    obs_ax.set_xlim(0, true_obs.shape[1])
    obs_ax.legend(loc="upper right")
    obs_ax.set_title("Predicted vs true observation timeseries")
    obs_ax.set_ylabel("Observation Value")

    act_ax.set_xlim(0, true_obs.shape[1])
    act_ax.legend(loc="upper right")
    act_ax.set_title("GRU unit activation timeseries")
    act_ax.set_xlabel("Time Step")
    act_ax.set_ylabel("Activation")

    # Plot phase portraits for true latents and GRU activations.
    true_phase_ax.plot(
        true_latents[0, :, 0].cpu().numpy(),
        true_latents[0, :, 1].cpu().numpy(),
    )
    true_phase_ax.set_title("True latent phase portrait")
    true_phase_ax.set_xlabel("latent x")
    true_phase_ax.set_ylabel("latent v")

    gru_phase_ax.plot(
        gru_activity[0, :, 0].cpu().numpy(),
        gru_activity[0, :, 1].cpu().numpy(),
    )
    gru_phase_ax.set_title("GRU latent phase portrait")
    gru_phase_ax.set_xlabel("gru unit 0")
    gru_phase_ax.set_ylabel("gru unit 1")

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
