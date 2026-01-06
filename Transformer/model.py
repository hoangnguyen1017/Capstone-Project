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
        # x: (B, T, D)
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)

class ViTFallDetector(nn.Module):
    """
    Input: (B, T, J, C) = (B, T, 14, 3)
    """

    def __init__(
        self,
        joints=14,
        channels=3,
        embed_dim=256,
        heads=4,
        layers=3,
        classes=2,
        max_len=64,
        dropout=0.3
    ):
        super().__init__()

        self.input_dim = joints * channels  

        self.pose_embed = nn.Linear(self.input_dim, embed_dim)
        self.vel_embed  = nn.Linear(self.input_dim, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02
        )
        encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
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
        """
        x: (B, T, 14, 3)
        """
        B, T, J, C = x.shape
        assert J * C == self.input_dim

        x = x.view(B, T, J * C)  
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]
        x = self.pose_embed(x) + self.vel_embed(vel)
        x = self.embed_dropout(x)
        x = x + self.pos_embed[:, :T]
        x = self.transformer(x)
        x = self.norm(x)
        feat = self.attn_pool(x)

        return self.fc(feat)
