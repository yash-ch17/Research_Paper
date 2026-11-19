import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradientReversal.apply(x, lambd)


class FrequencyBranch(nn.Module):
    """FFT magnitude -> compact CNN frequency representation."""

    def __init__(self, out_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x is ImageNet-normalized RGB. FFT is applied spatially.
        spec = torch.fft.fft2(x, dim=(-2, -1))
        mag = torch.log1p(torch.abs(spec))
        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        mag = (mag - mean) / std
        return self.proj(self.net(mag))


class GatedFusion(nn.Module):
    def __init__(self, rgb_dim, freq_dim, fusion_dim=384):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(rgb_dim + freq_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

    def forward(self, rgb, freq):
        r = self.rgb_proj(rgb)
        f = self.freq_proj(freq)
        g = self.gate(torch.cat([rgb, freq], dim=1))
        return self.out(torch.cat([r, g * f], dim=1))


class DeepfakeFusionModel(nn.Module):
    """
    RGB Xception + FFT CNN + gated fusion + BiGRU
    + classification head + adversarial domain head.
    """

    def __init__(
        self,
        backbone_name="xception",
        pretrained=True,
        fusion_dim=384,
        num_domains=7,
        gru_hidden=192,
        gru_layers=1,
        dropout=0.30,
    ):
        super().__init__()

        self.rgb = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        rgb_dim = self.rgb.num_features

        self.freq = FrequencyBranch(out_dim=192)
        self.fusion = GatedFusion(rgb_dim, 192, fusion_dim)

        self.temporal = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if gru_layers == 1 else dropout,
        )

        temporal_dim = gru_hidden * 2

        self.embedding = nn.Sequential(
            nn.Linear(temporal_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.classifier = nn.Linear(256, 2)

        self.domain_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_domains),
        )

    def forward(self, frames, grl_lambda=0.0):
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)

        rgb = self.rgb(x)
        freq = self.freq(x)

        fused = self.fusion(rgb, freq).reshape(b, t, -1)

        temporal, _ = self.temporal(fused)

        # Mean pooling is stable and matches common video-level aggregation.
        video_feature = temporal.mean(dim=1)

        embedding = self.embedding(video_feature)
        logits = self.classifier(embedding)

        domain_logits = self.domain_head(
            grad_reverse(embedding, grl_lambda)
        )

        return {
            "logits": logits,
            "embedding": embedding,
            "frame_features": fused,
            "domain_logits": domain_logits,
        }
