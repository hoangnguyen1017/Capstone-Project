import torch
import torch.nn as nn
import torch.nn.functional as F
from Preprocess import EDGES

def adjacency_matrix(v: int, edges):
    A = torch.zeros(v, v)
    idx_u = torch.tensor([u for u, v in edges])
    idx_v = torch.tensor([v for u, v in edges])
    A[idx_u, idx_v] = 1.0
    return A

V = 14
A_default = torch.as_tensor(adjacency_matrix(V, EDGES), dtype=torch.float32)

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, V=14, bias=True):
        super().__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

        self.B = nn.Parameter(torch.zeros(V, V))
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.ln = nn.GroupNorm(1, in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.register_buffer("_I", torch.eye(V))
        self.V = V

    def forward(self, x, A):
        B, C, T, Vn = x.shape
        device = x.device

        x = self.ln(x)

        A = A.to(device, x.dtype)
        A_tilde = A + self._I[:Vn, :Vn]

        deg = A_tilde.sum(1).clamp(min=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        B_norm = torch.softmax(self.B[:Vn, :Vn], dim=1)

        A_multi = A_norm + 0.5 * (A_norm @ A_norm)
        A_final = A_multi + self.alpha * B_norm
        A_final = A_final.unsqueeze(0)

        x = self.conv1x1(x)
        out = torch.einsum("bctv,bvw->bctw", x, A_final)

        return self.relu(out)
class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_sizes=[5,9,13], stride_t=1):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=(k,1),
                stride=(stride_t,1),
                padding=((k-1)//2, 0),
                groups=in_channels,
                bias=False
            ) for k in kernel_sizes
        ])

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        mid = max(out_channels // 8, 1)
        self.temporal_conv1 = nn.Conv1d(out_channels, mid, 1)
        self.temporal_conv2 = nn.Conv1d(mid, out_channels, 1)
        self.temporal_act = nn.ReLU()
        self.temporal_sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = None
        for conv in self.convs:
            y = conv(x)
            out = y if out is None else out + y

        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)

        att_in = out.mean(dim=-1)
        att = self.temporal_conv1(att_in)
        att = self.temporal_act(att)
        att = self.temporal_conv2(att)
        att = self.temporal_sigmoid(att).unsqueeze(-1)

        return out * att

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A,
                 kernel_sizes=[5,9,13], stride_t=1):
        super().__init__()

        self.gconv = GraphConv(
            in_channels, out_channels,
            V=A.shape[0] if hasattr(A, "shape") else V
        )

        self.tconv = MultiScaleTemporalConv(
            out_channels, out_channels,
            kernel_sizes, stride_t
        )

        self.A = A

        if in_channels != out_channels or stride_t != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=(stride_t,1), bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gconv(x, self.A)
        x = self.tconv(x)
        return self.relu(x + res)

class STGCNStream(nn.Module):
    def __init__(self, in_channels, A,
                 channels=[64,128,256],
                 kernel_sizes=[5,9,13]):
        super().__init__()

        layers = []
        c_in = in_channels

        for i, c_out in enumerate(channels):
            stride_t = 2 if i == len(channels) - 1 else 1
            layers.append(
                STGCNBlock(
                    c_in, c_out, A,
                    kernel_sizes, stride_t
                )
            )
            c_in = c_out

        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x):
        return self.net(x)

class AdaptiveAggregator(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()

        total_ch = sum(in_channels_list)

        self.se = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, max(ch//16,1), 1),
                nn.ReLU(),
                nn.Conv2d(max(ch//16,1), ch, 1),
                nn.Sigmoid()
            ) for ch in in_channels_list
        ])

        self.fc = nn.Linear(total_ch, 128)
        self.weight_fc = nn.Linear(128, 3)
        self.combine_fc = nn.Linear(total_ch, 256)

    def forward(self, feats):
        feats = [f * se(f) for f, se in zip(feats, self.se)]

        pooled = [
            f.mean(dim=[2,3]) + f.amax(dim=[2,3])
            for f in feats
        ]

        concat = torch.cat(pooled, dim=1)
        h = F.relu(self.fc(concat))
        w = F.softmax(self.weight_fc(h), dim=1)

        fused = torch.cat([
            pooled[i] * w[:, i:i+1] for i in range(3)
        ], dim=1)

        out = F.relu(self.combine_fc(fused))
        return out, w

class STGCNThreeStream(nn.Module):
    def __init__(self, num_class=2, A=None,
                 in_channels_each=3,
                 channels=[64,128,256]):
        super().__init__()

        if A is None:
            A = A_default
        if not isinstance(A, torch.Tensor):
            A = torch.as_tensor(A, dtype=torch.float32)

        self.joint_stream  = STGCNStream(in_channels_each, A, channels)
        self.bone_stream   = STGCNStream(in_channels_each, A, channels)
        self.motion_stream = STGCNStream(in_channels_each, A, channels)

        out_chs = [
            self.joint_stream.out_channels,
            self.bone_stream.out_channels,
            self.motion_stream.out_channels
        ]

        self.aggregator = AdaptiveAggregator(out_chs)

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_class)
        )

    def forward(self, x_joint, x_bone, x_motion):
        f1 = self.joint_stream(x_joint)
        f2 = self.bone_stream(x_bone)
        f3 = self.motion_stream(x_motion)

        fused, weights = self.aggregator([f1, f2, f3])
        logits = self.classifier(fused)

        return logits, weights
