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
    def __init__(self, in_channels, out_channels, V=14, bias=True, use_conf_mask=False):
        super().__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

        self.B = nn.Parameter(torch.zeros(V, V))
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.ln = nn.GroupNorm(1, in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.register_buffer("_I", torch.eye(V))
        self.V = V

        self.use_dynamic = use_conf_mask and (in_channels == 3)

        self.use_conf_mask = use_conf_mask and (in_channels == 3)

        if self.use_dynamic:
            self.dynamic_fc = nn.Linear(in_channels, V)
            self.gamma = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, A):
        B, C, T, Vn = x.shape
        device = x.device

        x = self.ln(x)

        if self.use_conf_mask:
            conf = x[:, -1:].detach()    # (B,1,T,V)
            x = x * conf

        A = A.to(device, x.dtype)
        A_tilde = A + self._I[:Vn, :Vn]
        deg = A_tilde.sum(1).clamp(min=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        B_norm = torch.softmax(self.B[:Vn, :Vn], dim=1)

        dyn = None
        if self.use_dynamic:
            x_mean = x.mean(2)
            x_embed = self.dynamic_fc(x_mean.transpose(1,2))
            dyn = torch.softmax(x_embed, dim=-1)

        A_multi = A_norm + 0.5 * (A_norm @ A_norm)

        A_final = A_multi + self.alpha * B_norm
        A_final = A_final.unsqueeze(0)

        if dyn is not None:
            A_final = A_final + self.gamma * dyn

        x = self.conv1x1(x)
        out = torch.einsum("bctv,bvw->bctw", x, A_final)
        return self.relu(out)

class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[5,7,9], stride_t=1):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k-1)//2
            self.convs.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=(k,1), stride=(stride_t,1),padding=(pad,0), groups=in_channels, bias=False))

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        mid = max(out_channels//8,1)
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
        out = out * att
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_sizes=[5,7,9], stride_t=1, use_conf_mask=True):
        super().__init__()
        self.gconv = GraphConv(in_channels, out_channels, V=A.shape[0] if hasattr(A,'shape') else V, use_conf_mask=use_conf_mask)
        self.tconv = MultiScaleTemporalConv(out_channels, out_channels, kernel_sizes, stride_t)
        self.A = A

        if in_channels != out_channels or stride_t != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride_t,1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gconv(x, self.A)
        x = self.tconv(x)
        x = x + res
        return self.relu(x)

class STGCNStream(nn.Module):
    def __init__(self, in_channels, A, channels=[128,256,512], kernel_sizes=[5,7,9],use_conf_mask=True):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            stride_t = 2 if i==len(channels)-1 else 1
            layers.append(STGCNBlock(c_in, c_out, A, kernel_sizes, stride_t,use_conf_mask=use_conf_mask))
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x):
        return self.net(x)


class AdaptiveAggregator(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        total_ch = sum(in_channels_list)

        self.fc = nn.Linear(total_ch, 256)
        self.classifier_pre = nn.Linear(256, 3)
        self.combine_fc = nn.Linear(total_ch, 512)

        # SE block cho từng stream
        self.se = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, max(ch // 16, 1), 1),
                nn.ReLU(),
                nn.Conv2d(max(ch // 16, 1), ch, 1),
                nn.Sigmoid()
            ) for ch in in_channels_list
        ])

    def forward(self, feats):

        feats_se = [f * se(f) for f, se in zip(feats, self.se)]
        pooled = []
        for f in feats_se:
            mean_p = f.mean(dim=[2,3])
            max_p  = f.amax(dim=[2,3])
            pooled.append(mean_p + max_p)
        
        concat = torch.cat(pooled, dim=1)

        h = F.relu(self.fc(concat))
        w_logits = self.classifier_pre(h)
        weights = F.softmax(w_logits, dim=1)

        w0 = weights[:, 0:1]
        w1 = weights[:, 1:2]
        w2 = weights[:, 2:3]

        fused = torch.cat([
            pooled[0] * w0,
            pooled[1] * w1,
            pooled[2] * w2
        ], dim=1)

        out = F.relu(self.combine_fc(fused))
        return out, weights

class STGCNThreeStream(nn.Module):
    def __init__(self, num_class=2, A=None, in_channels_each=3, channels=[128,256,512]):
        super().__init__()
        if A is None:
            A = A_default
        if not isinstance(A, torch.Tensor):
            A = torch.as_tensor(A, dtype=torch.float32)

        self.joint_stream = STGCNStream(in_channels_each, A, channels,use_conf_mask=True)
        self.bone_stream = STGCNStream(in_channels_each, A, channels,use_conf_mask=True)
        self.motion_stream = STGCNStream(in_channels_each, A, channels,use_conf_mask=False)

        out_chs = [
            self.joint_stream.out_channels,
            self.bone_stream.out_channels,
            self.motion_stream.out_channels
        ]

        self.aggregator = AdaptiveAggregator(out_chs)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,num_class)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == 1:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.out_features > 3:  
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        last_fc = self.classifier[-1]
        nn.init.constant_(last_fc.weight, 0)
        nn.init.constant_(last_fc.bias, 0)

    def forward(self, x_joint, x_bone, x_motion):

        f1 = self.joint_stream(x_joint)
        f2 = self.bone_stream(x_bone)
        f3 = self.motion_stream(x_motion)

        fused, weights = self.aggregator([f1,f2,f3])
        logits = self.classifier(fused)
        return logits, weights
