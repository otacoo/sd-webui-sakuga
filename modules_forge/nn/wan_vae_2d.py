import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config

from modules_forge.nn._vae import ProcessLatent


class RMSNorm2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=1).to(x) * self.scale * self.gamma.to(x)


def _make_residual_seq(in_dim, out_dim, dropout):
    return nn.Sequential(
        RMSNorm2D(in_dim),
        nn.SiLU(),
        nn.Conv2d(in_dim, out_dim, 3, padding=1),
        RMSNorm2D(out_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Conv2d(out_dim, out_dim, 3, padding=1),
    )


class QwenImageAttentionBlock2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        k = k.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        v = v.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).transpose(1, 2).reshape(b, c, h, w)
        x = self.proj(x)
        return x + identity


class QwenImageEncoder2D(nn.Module):
    def __init__(self, dim=96, z_dim=32, input_channels=3, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=[], dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, dim, 3, padding=1)
        self.downsamples = nn.ModuleList()
        dims = [dim * m for m in [1] + list(dim_mult)]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                entry = nn.ModuleDict({"residual": _make_residual_seq(in_dim, out_dim, dropout)})
                if in_dim != out_dim:
                    entry["shortcut"] = nn.Conv2d(in_dim, out_dim, 1)
                self.downsamples.append(entry)
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                self.downsamples.append(nn.ModuleDict({
                    "resample": nn.Sequential(
                        nn.ZeroPad2d((0, 1, 0, 1)),
                        nn.Conv2d(out_dim, out_dim, 3, stride=2),
                    ),
                }))
        self.middle = nn.ModuleList([
            nn.ModuleDict({"residual": _make_residual_seq(out_dim, out_dim, dropout)}),
            nn.ModuleDict({
                "norm": RMSNorm2D(out_dim),
                "to_qkv": nn.Conv2d(out_dim, out_dim * 3, 1),
                "proj": nn.Conv2d(out_dim, out_dim, 1),
            }),
            nn.ModuleDict({"residual": _make_residual_seq(out_dim, out_dim, dropout)}),
        ])
        self.head = nn.Sequential(
            RMSNorm2D(out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for entry in self.downsamples:
            if "residual" in entry:
                shortcut_out = entry["shortcut"](x) if "shortcut" in entry else x
                x = shortcut_out + entry["residual"](x)
            if "resample" in entry:
                x = entry["resample"](x)
        for entry in self.middle:
            if "residual" in entry:
                x = x + entry["residual"](x)
            if "to_qkv" in entry:
                res = x
                x = entry["norm"](x)
                q, k, v = entry["to_qkv"](x).chunk(3, dim=1)
                b, c, h, w = q.shape
                q = q.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
                k = k.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
                v = v.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
                x = F.scaled_dot_product_attention(q, k, v)
                x = x.squeeze(1).transpose(1, 2).reshape(b, c, h, w)
                x = entry["proj"](x)
                x = x + res
        return self.head(x)


class QwenImageDecoder2D(nn.Module):
    def __init__(self, dim=96, z_dim=16, output_channels=3, dim_mult=(1, 2, 4, 4), num_res_blocks=2, dropout=0.0):
        super().__init__()
        dims = [dim * m for m in [1] + list(dim_mult)]
        rev_dims = list(reversed(dims))
        max_dim = rev_dims[0]

        self.conv1 = nn.Conv2d(z_dim, max_dim, 3, padding=1)
        self.middle = nn.ModuleList([
            nn.ModuleDict({"residual": _make_residual_seq(max_dim, max_dim, dropout)}),
            nn.ModuleDict({
                "norm": RMSNorm2D(max_dim),
                "to_qkv": nn.Conv2d(max_dim, max_dim * 3, 1),
                "proj": nn.Conv2d(max_dim, max_dim, 1),
            }),
            nn.ModuleDict({"residual": _make_residual_seq(max_dim, max_dim, dropout)}),
        ])

        stage_outputs = [dim * m for m in reversed(dim_mult)]
        n_stages = len(stage_outputs)
        n_res = num_res_blocks + 1
        stage_first_in = [stage_outputs[0]]
        for s in range(1, n_stages):
            prev = stage_outputs[s - 1]
            next_in = prev
            for v in stage_outputs[s:]:
                if v != prev:
                    next_in = v
                    break
            stage_first_in.append(next_in)

        self.upsamples = nn.ModuleList()
        for s in range(n_stages):
            out_c = stage_outputs[s]
            in_c = stage_first_in[s]
            for _ in range(n_res):
                entry = nn.ModuleDict({"residual": _make_residual_seq(in_c, out_c, dropout)})
                if in_c != out_c:
                    entry["shortcut"] = nn.Conv2d(in_c, out_c, 1)
                self.upsamples.append(entry)
                in_c = out_c
            if s < n_stages - 1:
                next_in = stage_first_in[s + 1]
                self.upsamples.append(nn.ModuleDict({
                    "resample": nn.Sequential(
                        nn.Upsample(scale_factor=2.0, mode="nearest-exact"),
                        nn.Conv2d(out_c, next_in, 3, padding=1),
                    ),
                }))

        last_c = stage_outputs[-1]
        self.conv_out = nn.Conv2d(last_c, dim, 1) if last_c != dim else nn.Identity()
        self.head = nn.Sequential(
            RMSNorm2D(dim),
            nn.SiLU(),
            nn.Conv2d(dim, output_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for entry in self.middle:
            if "residual" in entry:
                x = x + entry["residual"](x)
            if "to_qkv" in entry:
                res = x
                x = entry["norm"](x)
                q, k, v = entry["to_qkv"](x).chunk(3, dim=1)
                b, c, h, w = q.shape
                q = q.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
                k = k.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
                v = v.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
                x = F.scaled_dot_product_attention(q, k, v)
                x = x.squeeze(1).transpose(1, 2).reshape(b, c, h, w)
                x = entry["proj"](x)
                x = x + res
        for entry in self.upsamples:
            if "residual" in entry:
                shortcut_out = entry["shortcut"](x) if "shortcut" in entry else x
                x = shortcut_out + entry["residual"](x)
            if "resample" in entry:
                x = entry["resample"](x)
        x = self.conv_out(x)
        return self.head(x)


class Qwen2DVAE(nn.Module, ProcessLatent, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(self, base_dim=96, z_dim=16, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=[], image_channels=3, dropout=0.0):
        super().__init__()
        self.base_dim = base_dim
        self.z_dim = z_dim
        self.latent_channels = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)

        self.encoder = QwenImageEncoder2D(
            dim=base_dim,
            z_dim=z_dim * 2,
            input_channels=image_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            dropout=dropout,
        )
        self.conv1 = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder2D(
            dim=base_dim,
            z_dim=z_dim,
            output_channels=image_channels,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

    def _flatten_frames(self, x: torch.Tensor) -> tuple[torch.Tensor, object]:
        if x.ndim == 4:
            return x, None
        if x.ndim != 5:
            raise ValueError(f"Unsupported Qwen2D VAE input shape: {tuple(x.shape)}")
        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        return x, (batch, frames)

    def _restore_frames(self, x: torch.Tensor, frame_info) -> torch.Tensor:
        if frame_info is None:
            return x
        batch, frames = frame_info
        channels, height, width = x.shape[1:]
        return x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, frame_info = self._flatten_frames(x)
        moments = self.conv1(self.encoder(x))
        mu, _ = moments.chunk(2, dim=1)
        return self._restore_frames(mu, frame_info)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z, frame_info = self._flatten_frames(z)
        out = self.decoder(self.conv2(z))
        out = torch.clamp(out, min=-1.0, max=1.0)
        return self._restore_frames(out, frame_info)
