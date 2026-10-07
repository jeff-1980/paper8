"""Sample-adaptive gated early fusion (MST revision, round 3).

Replaces the shared Conv1d embedding of a learner by one embedding branch per
channel; a small gate computes, for every sample, softmax weights over the two
branches from statistics of the learned branch features (mean of GELU(h) and
standard deviation over time), and the weighted sum of the branch features is
passed to the unchanged backbone. Unlike the earlier channel-attention variant,
the gate input depends on the sample even though windows are z-scored per
channel (the std over time of a learned linear filter output, and the mean of a
non-linear function of it, are not fixed by z-scoring).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionEmbed(nn.Module):
    def __init__(self, d_model, kernel_size, stride, padding, n_sensors=2, hidden=32):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(1, d_model, kernel_size=kernel_size, stride=stride, padding=padding)
            for _ in range(n_sensors)])
        self.gate = nn.Sequential(nn.Linear(2 * d_model * n_sensors, hidden), nn.GELU(),
                                  nn.Linear(hidden, n_sensors))
        self.last_gates = None

    @property
    def weight(self):            # BearMamba3.forward casts its input to conv_embed.weight.dtype
        return self.branches[0].weight

    def forward(self, x):        # x: (B, C, T)
        hs = [b(x[:, c:c + 1].to(b.weight.dtype)) for c, b in enumerate(self.branches)]   # each (B, d, L)
        stats = [torch.cat([F.gelu(h).mean(-1), h.std(-1)], dim=1) for h in hs]
        g = torch.softmax(self.gate(torch.cat(stats, dim=1).float()), dim=-1)          # (B, C)
        self.last_gates = g.detach()
        h = sum(g[:, c, None, None].to(hs[c].dtype) * hs[c] for c in range(len(hs)))
        return h


def make_gated(model, d_model, kernel_size, stride, padding, n_sensors=2):
    model.conv_embed = GatedFusionEmbed(d_model, kernel_size, stride, padding, n_sensors)
    return model


class TokenGatedFusionEmbed(nn.Module):
    """Post hoc variant (added after the window-level gate failed its adaptivity check):
    per-token gate g_t = softmax(MLP([h1_t, h2_t])) over channels, so the fusion weight can vary
    within and across windows."""
    def __init__(self, d_model, kernel_size, stride, padding, n_sensors=2, hidden=32):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(1, d_model, kernel_size=kernel_size, stride=stride, padding=padding)
            for _ in range(n_sensors)])
        self.gate = nn.Sequential(nn.Linear(d_model * n_sensors, hidden), nn.GELU(),
                                  nn.Linear(hidden, n_sensors))
        self.last_gates = None          # (B, C) window-mean gate
        self.last_token_gates = None    # (B, L, C)

    @property
    def weight(self):
        return self.branches[0].weight

    def forward(self, x):
        hs = [b(x[:, c:c + 1].to(b.weight.dtype)) for c, b in enumerate(self.branches)]   # (B, d, L)
        z = torch.cat([h.transpose(1, 2) for h in hs], dim=-1).float()                   # (B, L, C*d)
        g = torch.softmax(self.gate(z), dim=-1)                                            # (B, L, C)
        self.last_token_gates = g.detach(); self.last_gates = g.detach().mean(1)
        return sum(g[..., c].unsqueeze(1).to(hs[c].dtype) * hs[c] for c in range(len(hs)))


def make_token_gated(model, d_model, kernel_size, stride, padding, n_sensors=2):
    model.conv_embed = TokenGatedFusionEmbed(d_model, kernel_size, stride, padding, n_sensors)
    return model
