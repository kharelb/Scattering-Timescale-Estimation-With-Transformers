import torch
from torch import nn
import torch.nn.functional as F


class MultimodelTransformerEncoder(nn.Module):
    """Encode dynamic spectrum and time-series streams with separate Transformers.

    Both inputs are passed through their own nn.TransformerEncoder and
    the resulting token-wise embeddings are concatenated along the
    feature dimension (last axis). The module assumes `batch_first=True`
    for the internal Transformer layers.

    Inputs:
    - ds: Tensor of shape [B, T_img, D]
    - ts:  Tensor of shape [B, T_ts, D]

    Output:
    - concatenated: Tensor with same time dimension and feature dim = 2*D
    """

    def __init__(self, d_model, num_layers, nhead):
        super().__init__()
        img_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, batch_first=True, nhead=nhead
        )
        self.img_transformer = nn.TransformerEncoder(img_encoder, num_layers=num_layers)
        ts_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, batch_first=True, nhead=nhead
        )
        self.ts_transformer = nn.TransformerEncoder(ts_encoder, num_layers=num_layers)

    def forward(self, img, ts):
        img_encoded = self.img_transformer(img)
        ts_encoded = self.ts_transformer(ts)

        concatenated = torch.concat((img_encoded, ts_encoded), dim=-1)

        return concatenated


class AttentionPooling(nn.Module):
    """Learned attention pooling over the temporal dimension.

    Computes a per-timestep score with a single linear layer, applies a
    softmax over the time axis to obtain weights, and returns the
    weighted sum across time as a fixed-size vector per batch.

    Input: `x` of shape [B, T, D]
    Output: `v` of shape [B, D]
    """

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):  # z: [B, T, D] - output from transformer
        scores = self.attn(x).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=1)  # attention weights α_t
        v = torch.sum(x * weights.unsqueeze(-1), dim=1)  # weighted sum -> [B, D]
        return v


class HurdleRegressionHead(nn.Module):
    """
    Zero-inflated (hurdle) head for positive-valued regression targets.
    Predicts:
        p0  = P(target == 0)
        mu  = mean of log(target) | target > 0
        sigma = std of log(target) | target > 0
    """

    def __init__(self, d_model, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.p0_head = nn.Linear(hidden_dim, 1)  # probability of zero scattering
        self.mu_head = nn.Linear(hidden_dim, 1)  # mean of log(τ)
        self.s_head = nn.Linear(hidden_dim, 1)  # std of log(τ)

    def forward(self, v):  # v: [B, D] (e.g., output of attention pooling)
        h = self.shared(v)
        p0 = torch.sigmoid(self.p0_head(h))  # [B, 1]
        mu = self.mu_head(h)  # [B, 1]
        sigma = F.softplus(self.s_head(h)) + 1e-6  # ensure σ > 0
        return p0, mu, sigma


class TransformerBasedRegressor(nn.Module):
    """End-to-end transformer-based regressor for zero-inflated targets.

    The model projects scalar time-series values into `d_model`-dim
    embeddings, adds positional embeddings, encodes the image and
    time-series streams with `MultimodelTransformerEncoder`, applies
    attention pooling, and finally predicts a zero-probability and
    log-normal parameters via `HurdleRegressionHead`.

    Expected `cfg` keys: ``d_model``, ``drop_rate``, ``context_length``,
    ``num_layers``, ``nhead``.

    Forward inputs:
    - img: image-like tensor (batch, ...). Internally reshaped to
        [B, T_img, D].
    - ts:  time-series tensor (batch, ...). Internally reshaped to
        [B, T_ts, 1].

    Returns tuple `(p0, mu, sigma)` with shapes [B, 1] each.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ts_proj = nn.Linear(1, cfg["d_model"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["d_model"])

        self.layer_norm_img = nn.LayerNorm(cfg["d_model"])
        self.layer_norm_ts = nn.LayerNorm(cfg["d_model"])

        self.transformer = MultimodelTransformerEncoder(
            cfg["d_model"], cfg["num_layers"], cfg["nhead"]
        )

        self.final_norm = nn.LayerNorm(cfg["d_model"] * 2)

        self.final_proj = nn.Linear(cfg["d_model"] * 2, cfg["d_model"])

        self.attn_pooling = AttentionPooling(cfg["d_model"])

        self.regression_head = HurdleRegressionHead(d_model=cfg["d_model"])

    def forward(self, img, ts):
        encoded_img = img.squeeze(1).transpose(1, 2)
        encoded_ts = ts.squeeze(1).unsqueeze(-1)
        ts_proj = self.ts_proj(encoded_ts)

        encoded_ts = ts_proj + self.pos_emb(
            torch.arange(ts_proj.shape[1], device=ts.device)
        )
        encoded_img = encoded_img + self.pos_emb(
            torch.arange(encoded_img.shape[1], device=img.device)
        )

        encoded_img = self.drop_emb(encoded_img)
        encoded_ts = self.drop_emb(encoded_ts)

        encoded_img = self.layer_norm_img(encoded_img)
        encoded_ts = self.layer_norm_ts(encoded_ts)

        x = self.transformer(encoded_img, encoded_ts)

        x = self.final_norm(x)

        x = self.final_proj(x)

        x = self.attn_pooling(x)

        p0, mu, sigma = self.regression_head(x)

        return p0, mu, sigma
