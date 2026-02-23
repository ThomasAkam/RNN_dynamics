import torch
from torch import nn

# GRU predictor

class GRUPredictor(nn.Module):
    """GRU-based predictor for one-step-ahead forecasting of multivariate time series.
    Input and/or output can be optionally transformed with a ReLU MLP.  To use linear 
    input/output mapping, set embed_dim / decode_dim to None.
    """
    def __init__(self, obs_dim, hidden_size, embed_dim=8, decode_dim=8):
        """Args:
            obs_dim: dimensionality of observed vector at each time step
            hidden_size: number of GRU hidden units
            embed_dim: dimension of ReLU embedding layer for inputs, 
                       set to None for linear input mapping.
            decode_dim: dimension of ReLU layer for GRU output transformation,
                        set to None for linear output mapping."""
        super().__init__()
        if embed_dim is None: # Linear input mapping.
            self.input_map = nn.Identity()
            self.gru = nn.GRU(obs_dim, hidden_size, batch_first=True)
        else: # Embed input using a ReLU layer.
            self.input_map = nn.Sequential(nn.Linear(obs_dim, embed_dim), nn.ReLU())
            self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        if decode_dim is None: # Linear output mapping.
            self.output_map = nn.Linear(hidden_size, obs_dim)
        else: # Transform GRU output with a ReLU layer.
            self.output_map = nn.Sequential(
                nn.Linear(hidden_size, decode_dim),
                nn.ReLU(),
                nn.Linear(decode_dim, obs_dim),
            )

    def forward(self, x, return_activations=False):
        embedded = self.input_map(x[:, :-1])
        gru_act, _ = self.gru(embedded)
        preds = self.output_map(gru_act)
        if return_activations:
            return preds, gru_act
        return preds