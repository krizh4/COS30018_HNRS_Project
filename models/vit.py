"""A very small Vision Transformer (Week 8) for 28x28 digits

The picture is cut into 16 patches of 7x7 pixels. Each patch becomes a vector,
a learned position vector is added, and transformer encoder layers let every
patch "attend" to every other patch. A special (CLS) vector collects the answer
"""
import torch
from torch import nn


class TinyViT(nn.Module):
    def __init__(self, patch=7, dim=64, depth=4, heads=4):
        super().__init__()
        n_patches = (28 // patch) ** 2                      # 16
        # A convolution with stride = kernel size is the usual trick to cut + embed patches
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 2,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch_embed(x)                    # (batch, dim, 4, 4)
        x = x.flatten(2).transpose(1, 2)           # (batch, 16, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))       # use the (CLS) vector to classify