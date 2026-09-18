import torch
import torch.nn as nn


class GatingLayer(nn.Module):
    """Gating mechanism for controlling information flow."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        combined = torch.cat([x, y], dim=-1)
        gate_weights = self.gate(combined)
        return gate_weights * x + (1 - gate_weights) * y


class TransformerBlock(nn.Module):
    """Self-attention block with feed-forward and residual connections."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class SimpleTFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer.
    Matches the architecture used in experiment_symbols_simple_tft.py.
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_heads: int = 8,
                 num_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_size))

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.variable_selection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size),
            nn.Softmax(dim=-1)
        )

        self.static_enrichment = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.gating_layer = GatingLayer(hidden_size)

        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        x = self.input_projection(x)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc

        # variable selection weights currently unused but kept for compatibility
        _ = self.variable_selection(x.mean(dim=1))

        for layer in self.transformer_layers:
            x = layer(x)

        static_context = x.mean(dim=1)
        static_enriched = self.static_enrichment(static_context)

        attn_output, _ = self.temporal_attention(x, x, x)
        x = self.gating_layer(x, attn_output)

        final_context = x[:, -1, :] + static_enriched
        output = self.output_layers(final_context)
        return output  # [batch_size, 1]


def create_simple_tft_model(input_size: int, hidden_size: int = 128, num_heads: int = 8,
                            num_layers: int = 3, dropout: float = 0.15) -> SimpleTFT:
    """Factory helper to build a SimpleTFT model."""
    return SimpleTFT(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
