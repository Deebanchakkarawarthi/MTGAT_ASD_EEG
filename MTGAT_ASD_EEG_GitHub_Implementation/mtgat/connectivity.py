import numpy as np
import torch
from torch import nn

DEFAULT_BANDS = {"delta": (1,4), "theta": (4,8), "alpha": (8,13),
                 "beta": (13,30), "gamma": (30,45)}

def bandpass_epoch(epoch, sfreq, low, high, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, np.asarray(epoch), axis=-1)

def plv_matrix(epoch, sfreq, bands=None):
    from scipy.signal import hilbert
    bands = bands or DEFAULT_BANDS
    result = []
    for low, high in bands.values():
        filtered = bandpass_epoch(epoch, sfreq, low, high)
        phase = np.angle(hilbert(filtered, axis=-1))
        diff = phase[:, None, :] - phase[None, :, :]
        plv = np.abs(np.mean(np.exp(1j * diff), axis=-1))
        np.fill_diagonal(plv, 1.0)
        result.append(plv.astype(np.float32))
    return np.stack(result)

class DifferentiableDynamicPLV(nn.Module):
    # Input phase [B,K,N,T]; output weighted synchrony [B,K,N,N].
    # Analytic phase extraction itself is performed upstream.
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, 1))
    def forward(self, phase):
        diff = phase.unsqueeze(3) - phase.unsqueeze(2)
        features = torch.stack((torch.cos(diff), torch.sin(diff)), dim=-1)
        weights = torch.softmax(self.scorer(features).squeeze(-1), dim=-1)
        return torch.sum(weights * torch.cos(diff), dim=-1).abs()
