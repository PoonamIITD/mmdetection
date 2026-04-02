import torch
import torch.nn as nn


class EncoderDecoderAdapter(nn.Module):
    """
    Lightweight adapter to align encoder embeddings before feeding
    into the DINO decoder (value input).
    """

    def __init__(self, embed_dims=256, bottleneck_dims=64):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        self.down = nn.Linear(embed_dims, bottleneck_dims)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dims, embed_dims)

        # initialize close to identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        """
        x: [bs, num_tokens, C]
        """
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return x + residual

class BottleneckAdapter(nn.Module):
    def __init__(self, embed_dims, reduction=4, init_scale=1e-3):
        super().__init__()
        bottleneck_dim = embed_dims // reduction

        self.norm = nn.LayerNorm(embed_dims)
        self.down = nn.Linear(embed_dims, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, embed_dims)

        # learnable scaling parameter
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x):
        # x: [bs, num_tokens, C]
        identity = x

        out = self.norm(x)
        out = self.down(out)
        out = self.act(out)
        out = self.up(out)

        return identity + self.scale * out
