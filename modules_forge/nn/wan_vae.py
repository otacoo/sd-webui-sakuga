"""3D Qwen-Image / Wan VAE used by Anima.

The Anima VAE checkpoint is a 3D (video) VAE.  Keeping its temporal
convolutions is required for correct image quality; converting it to a 2D
VAE loses color, detail, and text fidelity.  Only the single-frame path is
needed here, but the module names and parameter shapes match the checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from einops import rearrange

from modules_forge.nn._vae import ProcessLatent


class CausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self._causal_padding = tuple(padding)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, **kwargs)

    def forward(self, x):
        pt, ph, pw = self._causal_padding
        temporal_pad = max(0, self.kernel_size[0] - 1)
        if temporal_pad or ph or pw:
            x = F.pad(x, (pw, pw, ph, ph, temporal_pad, 0))
        return super().forward(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, images=True, bias=False):
        super().__init__()
        shape = (dim, 1, 1) if images else (dim, 1, 1, 1)
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        out = F.normalize(x, dim=1) * self.scale * self.gamma.to(x)
        if self.bias is not None:
            out = out + self.bias.to(x)
        return out


class Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        self.time_conv = None
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        b, c, t, h, w = x.shape
        if self.mode == "none":
            return x
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        return x + identity


class Encoder3d(nn.Module):
    def __init__(self, dim=96, z_dim=32, input_channels=3, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=(), temporal_downsample=(False, True, True), dropout=0.0):
        super().__init__()
        dims = [dim * m for m in [1] + list(dim_mult)]
        scale = 1.0

        self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        return self.head(x)


class Decoder3d(nn.Module):
    def __init__(self, dim=96, z_dim=16, output_channels=3, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=(), temporal_upsample=(False, True, True), dropout=0.0):
        super().__init__()
        dims = [dim * m for m in [dim_mult[-1]] + list(dim_mult)[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, output_channels, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        return self.head(x)


class WanVAE(nn.Module, ProcessLatent, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(self, base_dim=96, z_dim=16, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=(), temporal_downsample=(False, True, True), image_channels=3, conv_out_channels=3, dropout=0.0):
        super().__init__()
        self.dim = base_dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temporal_downsample = list(temporal_downsample)
        self.temporal_upsample = self.temporal_downsample[::-1]

        self.encoder = Encoder3d(
            dim=base_dim,
            z_dim=z_dim * 2,
            input_channels=image_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            temporal_downsample=self.temporal_downsample,
            dropout=dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim=base_dim,
            z_dim=z_dim,
            output_channels=conv_out_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            temporal_upsample=self.temporal_upsample,
            dropout=dropout,
        )

    def encode(self, x):
        moments = self.conv1(self.encoder(x))
        return moments.chunk(2, dim=1)[0]

    def decode(self, z):
        out = self.decoder(self.conv2(z))
        return torch.clamp(out, min=-1.0, max=1.0)
