import math

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SinusoidalPositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_t, h_patches, w_patches, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # # Generate 1D encodings for each axis
        d_t = d_model // 3
        d_h = d_model // 3
        # d_w = d_model // 3
        d_w = d_model - d_t - d_h  # Handle odd d_model by giving remaining channels to width

        pe_t = self._get_1d_pe(max_t, d_t)
        pe_h = self._get_1d_pe(h_patches, d_h)
        pe_w = self._get_1d_pe(w_patches, d_w)

        # Expand to full spatial-temporal grid
        pe_t = pe_t.view(max_t, 1, 1, d_t).expand(max_t, h_patches, w_patches, d_t)
        pe_h = pe_h.view(1, h_patches, 1, d_h).expand(max_t, h_patches, w_patches, d_h)
        pe_w = pe_w.view(1, 1, w_patches, d_w).expand(max_t, h_patches, w_patches, d_w)

        # Concatenate along feature dimension
        pe = torch.cat([pe_t, pe_h, pe_w], dim=-1)


        # Flatten spatial and temporal dimensions into the 1D sequence format
        pe = pe.reshape(1, max_t * h_patches * w_patches, d_model)
        self.register_buffer("pe", pe)

    def _get_1d_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # Ensure cosine terms match the available indices even if d_model is odd
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.dtype)
        return self.dropout(x)


class LearnablePositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_t, h_patches, w_patches, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_t = nn.Parameter(torch.randn(max_t, 1, 1, d_model) * 0.02)
        self.pe_h = nn.Parameter(torch.randn(1, h_patches, 1, d_model) * 0.02)
        self.pe_w = nn.Parameter(torch.randn(1, 1, w_patches, d_model) * 0.02)

    @property
    def pe(self):
        pe_3d = self.pe_t + self.pe_h + self.pe_w
        return pe_3d.reshape(1, -1, pe_3d.shape[-1])

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.dtype)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 20, dropout: float = 0.2):
        super().__init__()
        # the data is in shape (B, T, L)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.apply(self.init_weights)

    def init_weights(self, m):
        torch.nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :].to(x.dtype))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 20, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Using d_model as denominator for frequency scaling
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match the 3D pattern: [1, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :].to(x.dtype)
        return self.dropout(x)


class SpatialPositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=7, width=7, dropout=0.1):
        super().__init__()
        # Flattened spatial learnable embeddings: [1, H*W, D]
        self.pe = nn.Parameter(torch.randn(1, height * width, d_model) * 0.02)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x is [B, N, D] where N = height * width
        x = x + self.pe[:, : x.size(1), :].to(x.dtype)
        return self.dropout(x)


def get_positional_encoder(pos_type: str, is_3d: bool, pos_encoding_cfg: DictConfig, **kwargs):
    """
    Factory to create positional encoding modules.

    Args:
        pos_type (str): The type of positional encoding ('learnable', 'sinusoidal', 'learnable3d', 'sinusoidal3d').
        is_3d (bool): Flag to request a 3D positional encoder.
        pos_encoding_cfg (DictConfig): The configuration for the positional encoding module (e.g., dropout, d_model).
        **kwargs: Module-specific arguments like max_len, max_t, h_patches, w_patches.

    Returns:
        A positional encoding module instance.
    """
    if is_3d and pos_type in ["learnable3d", "sinusoidal3d"]:
        # Remove 1D-specific kwargs
        kwargs.pop('max_len', None)
        if pos_type == "learnable3d":
            return LearnablePositionalEncoding3D(**kwargs, **pos_encoding_cfg)
        else:
            return SinusoidalPositionalEncoding3D(**kwargs, **pos_encoding_cfg)

    # All other cases are 1D. This includes explicit 1D types,
    # and 3D types where is_3d=False.
    effective_pos_type = pos_type.replace("3d", "")

    # Remove 3D-specific kwargs if they exist
    kwargs.pop('max_t', None)
    kwargs.pop('h_patches', None)
    kwargs.pop('w_patches', None)

    if effective_pos_type == "learnable":
        return LearnablePositionalEncoding(**kwargs, **pos_encoding_cfg)
    elif effective_pos_type == "sinusoidal":
        return SinusoidalPositionalEncoding(**kwargs, **pos_encoding_cfg)
    else:
        raise TypeError(
            f"Provided positional encoding type '{pos_type}' is not supported for the requested dimension."
        )