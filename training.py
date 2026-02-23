import numpy as np
import torch


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
    """Train the GRU predictor model on the training observations and evaluate on validation data.
    Returns the final validation predictions and GRU activations."""
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
            optimizer.zero_grad()
            preds, gru_activity = model(batch, return_activations=True)
            target = batch[:, 1:, :]
            loss = loss_fn(preds, target)
            if reg_lambda > 0.0:  # Apply L2 regularization to GRU activations.
                loss += reg_lambda * gru_activity.pow(2).mean()
            if weight_decay > 0.0:  # Apply L2 regularization to model parameters.
                loss += weight_decay * sum(param.pow(2).sum() for param in model.parameters())
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
            print(f"Epoch {ep:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    return val_preds, val_activity
