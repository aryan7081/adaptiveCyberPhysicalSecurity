"""
Tabular Masked Autoencoder (MAE) with BERT-Style Masking
Pre-trains on benign traffic only. Learns robust representations for anomaly detection.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Mask token placeholder (will be replaced by learnable embedding in forward)
MASK_VALUE = -999.0  # Distinct from valid data after scaling


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding for 1D sequences (feature dimension)."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return x + self.pe[:, : x.size(1), :]


class TabularMAE(nn.Module):
    """
    Masked Autoencoder for tabular/sequence data with BERT-style masking.
    - Encoder: processes only unmasked positions (efficient)
    - Decoder: reconstructs masked values from latent + mask tokens
    - Mask ratio ~15% (BERT standard) for tabular
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        init: str = "xavier",
    ):
        super().__init__()
        self.num_features = num_features
        self.mask_ratio = mask_ratio
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(1, hidden_dim)  # Each feature value -> hidden
        self.mask_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_enc = PositionalEncoding1D(hidden_dim, max_len=num_features)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # Decoder: lightweight
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )
        self.reconstruct_head = nn.Linear(hidden_dim, 1)

        self._init_weights(init)

    def _init_weights(self, init: str):
        """Xavier/Glorot initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == "xavier":
                    nn.init.xavier_uniform_(m.weight, gain=0.1 if m == self.reconstruct_head else 1.0)
                else:
                    nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _create_mask(self, B: int, L: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        BERT-style random masking: mask ~mask_ratio of positions per sample.
        Returns: (mask_bool, mask_indices) where mask_bool marks masked positions.
        """
        num_masked = max(1, int(L * self.mask_ratio))
        # Different mask per sample
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i in range(B):
            idx = torch.randperm(L, device=device)[:num_masked]
            mask[i, idx] = True
        return mask, mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
        no_mask: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        x: [B, L] raw feature values (L = num_features)
        no_mask: if True, no masking (for reconstruction-based inference)
        Returns: (reconstruction_loss, embeddings, optional_decoder_output)
        """
        B, L = x.shape
        device = x.device

        if mask is None:
            if no_mask:
                mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            else:
                mask, _ = self._create_mask(B, L, device)

        # Project each feature to hidden_dim: [B, L, 1] -> [B, L, H]
        x_expand = x.unsqueeze(-1)
        tokens = self.input_proj(x_expand)

        # Replace masked positions with mask embedding (BERT-style)
        mask_emb = self.mask_embed.expand(B, L, -1)
        tokens = torch.where(mask.unsqueeze(-1), mask_emb, tokens)
        tokens = self.pos_enc(tokens)

        # Encode: Transformer sees all tokens (masked positions have mask embedding)
        encoded = self.encoder(tokens)
        encoded = self.encoder_norm(encoded)

        # Global embedding: mean over sequence (for downstream OCSVM)
        embedding = encoded.mean(dim=1)  # [B, H]

        if return_embedding:
            return embedding

        # Decoder: reconstruct masked positions only (efficient)
        mask_tokens = self.mask_embed.expand(B, L, -1).clone()
        mask_tokens = self.pos_enc(mask_tokens)
        decoded = self.decoder(mask_tokens, encoded)
        logits = self.reconstruct_head(decoded).squeeze(-1)  # [B, L]

        # Loss: on masked positions (BERT-style) or all positions (no_mask)
        target = x.detach()
        if no_mask or mask.sum() == 0:
            loss = F.mse_loss(logits, target)
        else:
            loss = F.mse_loss(logits[mask], target[mask])
        return loss, embedding, logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without masking (for inference)."""
        B, L = x.shape
        x_expand = x.unsqueeze(-1)
        tokens = self.input_proj(x_expand)
        tokens = self.pos_enc(tokens)
        encoded = self.encoder(tokens)
        encoded = self.encoder_norm(encoded)
        return encoded.mean(dim=1)
