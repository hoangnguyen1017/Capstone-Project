import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)


class ViTFallDetector(nn.Module):
    def __init__(
        self,
        input_dim=42,          
        embed_dim=256,
        heads=4,
        layers=3,
        classes=2,
        max_len=64,
        dropout=0.3
    ):
        super().__init__()

        self.input_dim = input_dim

        self.pose_embed = nn.Linear(input_dim, embed_dim)
        self.vel_embed  = nn.Linear(input_dim, embed_dim)

        self.embed_dropout = nn.Dropout(dropout)

        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02
        )

        encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder, layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_pool = AttentionPooling(embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        assert D == self.input_dim, f"Expected {self.input_dim}, got {D}"

        # velocity
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]

        x = self.pose_embed(x) + self.vel_embed(vel)
        x = self.embed_dropout(x)

        x = x + self.pos_embed[:, :T]
        x = self.transformer(x)
        x = self.norm(x)

        feat = self.attn_pool(x)
        return self.fc(feat)
