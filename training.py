"""Functions for training the GRU predictor model on data."""

import torch
from torch.utils.data import DataLoader


def train_model(
    model,
    train_obs,
    val_obs,
    epochs,
    batch_size,
    optimizer,
    loss_fn,
    reg_lambda=0.0,
    weight_decay=0.0,
):
    """Train the GRU predictor model and evaluate on validation data.

    Args:
        model: Sequence prediction model.
        train_obs: Training observation tensor of shape (n_sequences, seq_len, obs_dim).
        val_obs: Validation observation tensor of shape (m_sequences, seq_len, obs_dim).
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size used by the training DataLoader.
        optimizer: Torch optimizer used to update model parameters.
        loss_fn: Loss function applied to one-step-ahead predictions.
        reg_lambda: L2 penalty coefficient applied to GRU activations.
        weight_decay: L2 penalty coefficient applied to model parameters.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            ``(val_preds, val_activity)`` from the final validation forward pass.
    """
    prev_val_loss = None
    train_loader = DataLoader(train_obs, batch_size=batch_size, shuffle=True)

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            preds, gru_activity = model(batch, return_activations=True)
            target = batch[:, 1:, :]
            loss = loss_fn(preds, target)
            if reg_lambda > 0.0:  # Apply L2 regularization to GRU activations.
                loss += reg_lambda * gru_activity.pow(2).mean()
            if weight_decay > 0.0:  # Apply L2 regularization to model parameters.
                loss += weight_decay * sum(
                    param.pow(2).sum() for param in model.parameters()
                )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batches += 1
        train_loss /= batches

        model.eval()
        with torch.no_grad():
            val_preds, val_activity = model(val_obs, return_activations=True)
            val_loss = loss_fn(val_preds, val_obs[:, 1:, :]).item()

        if prev_val_loss is not None and val_loss > prev_val_loss:
            print(
                f"Early stopping at epoch {ep:3d}: "
                f"val_loss rose from {prev_val_loss:.6f} to {val_loss:.6f}"
            )
            break

        prev_val_loss = val_loss

        if ep % 5 == 0 or ep == 1:
            print(
                f"Epoch {ep:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

    return val_preds, val_activity
