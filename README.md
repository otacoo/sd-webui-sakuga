<h1 align="center">Stable Diffusion WebUI Sakuga</h1>

<p align="center">
  <b>A standalone, experimental fork of Forge Classic with native Anima support.</b>
</p>

> [!WARNING]
> Anima support is experimental and under active development. Expect breaking changes and rough edges. Sakuga is an independent fork and is not affiliated with or endorsed by the upstream Forge projects. Please report Sakuga issues to this repository, not to the upstream projects.

## What is Sakuga?

Sakuga is a focused Stable Diffusion WebUI built on the **Forge Classic** codebase. It keeps the lightweight SD1/SDXL workflow of Classic and adds native support for **Anima**, a text-to-image model based on the NVIDIA Cosmos architecture with a Qwen3-0.6B text encoder.

Current architectures: **SD1**, **SDXL**, and **Anima**. Other model families are not supported.

It is intended for users who want a compact WebUI, fast startup, and Anima support without the extra model families and bloat of larger forks.

## Features

### Anima

- Native Anima inference with flow matching
- Load Anima from either:
  - a single `.safetensors` checkpoint, or
  - an `AnimaPipeline` diffusers directory
- Qwen3-0.6B and T5 text processing, including prompt emphasis and `BREAK`
- `ER SDE` sampler for flow-based models
- 16-channel latent previews
- ControlNet-LLLite adapter for Anima *(experimental)*
- Text encoder selection and VAE compatibility fallback
- Bundled Qwen3 and T5 tokenizers — no separate tokenizer download required

### Forge Classic foundation

- SD1 and SDXL checkpoints
- LoRA, ControlNet, IP-Adapter, MultiDiffusion, soft inpainting, and the usual Classic extensions
- SageAttention, FlashAttention, `fp8`, persistent LoRA patching, and in-place LoRA patching
- Sampler and scheduler selection, including `ER SDE`
- Hires. fix CFG, MaHiRo, RescaleCFG, VAE tiling, and the other Classic optimizations

### Sakuga UI and tooling

- Compact prompt layout with the scheduler visible by default
- `reilnuud/polite` Gradio theme and larger gallery/preview layout
- Optional **Hide Styles** setting
- Live previews in the full-page image viewer
- Windows `install.bat`, `start-webui.bat`, and `update.bat` helpers

## Requirements

- **Windows 10/11** — primary target
- **Git**
- **Python 3.11.x** — 3.11.9 recommended, or use `uv`
  - On Windows, `webui.bat` automatically looks for Python 3.11 even if another Python version comes first on `PATH`. If yours is in a non-standard location, set `PYTHON` in `webui-user.bat`.
- An NVIDIA GPU with current CUDA drivers is strongly recommended
- Enough disk space for Python packages and model weights

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/otacoo/sd-webui-sakuga.git
   cd sd-webui-sakuga
   ```

2. Run the installer:

   ```bat
   install.bat
   ```

   The installer asks which optional packages and performance flags you want, then writes them to `start-webui.bat`. You can re-run it at any time to reconfigure.

3. Put your models in the folders described below.

4. Start the WebUI:

   ```bat
   start-webui.bat
   ```

   The first launch installs the Python requirements and may take a while. Later launches are much faster.

By default the launcher uses `--no-download-sd-model`, so the WebUI will start without a checkpoint and wait for you to add one.

## Model setup

Model weights are not included with Sakuga. Download the checkpoints and components you need from their respective sources.

Sakuga uses ComfyUI-style model folders. Checkpoints and LoRAs are scanned recursively, so you can place files directly in the parent folder or organize them by model family:

```text
models/
├─ diffusion_models/
│  ├─ SD15/
│  ├─ SDXL/
│  ├─ Illustrious/
│  ├─ Pony/
│  └─ Anima/
└─ loras/
   ├─ SD15/
   ├─ SDXL/
   ├─ Illustrious/
   ├─ Pony/
   └─ Anima/
```

`Illustrious` and `Pony` are SDXL-family models and use the same SDXL loading path; the separate folders are for organization only.

### SD1 and SDXL

Place SD1/SDXL checkpoints directly in `models/diffusion_models/` or in a model-family subfolder.

Other model types use the usual folders:

```text
models/loras/          LoRA and LyCORIS
models/VAE/            VAE files
models/embeddings/     textual inversions
models/ESRGAN/         upscalers
```

### Anima

Anima needs the transformer, the VAE, and the Qwen3-0.6B text encoder. The tokenizers are already bundled with Sakuga.

#### Option A — single-file checkpoint

```text
models/
├─ diffusion_models/
│  └─ Anima/
│     └─ anima.safetensors
├─ VAE/
│  └─ qwen_image_vae.safetensors
└─ text_encoders/
   └─ qwen_3_06b_base.safetensors
```

Create `models/text_encoders/` if it does not exist. The default filenames are `qwen_image_vae.safetensors` and `qwen_3_06b_base.safetensors`; you can select different files in Settings if needed.

#### Option B — diffusers directory

```text
models/diffusion_models/Anima/
├─ model_index.json
├─ transformer/
│  └─ diffusion_pytorch_model.safetensors
├─ vae/
│  └─ diffusion_pytorch_model.safetensors
└─ text_encoder/
   └─ diffusion_pytorch_model.safetensors   (optional)
```

`model_index.json` must contain:

```json
{
  "_class_name": "AnimaPipeline"
}
```

If `text_encoder/` is omitted, Sakuga uses `models/text_encoders/qwen_3_06b_base.safetensors`. Anima directories are detected automatically and appear in the checkpoint list.

### Using Anima

1. Select the Anima checkpoint in the model dropdown.
2. For single-file Anima models, choose a text encoder or VAE in Settings if you are not using the defaults.
3. Diffusers directories use the `text_encoder/` and `vae/` folders inside the model directory. If `text_encoder/` is missing, the external default is used.
4. A reasonable starting point is:

   | Setting | Value |
   | --- | --- |
   | Sampler | `ER SDE` |
   | Scheduler | `Beta` |
   | Steps | `32` |
   | CFG scale | `4.0` |

   Adjust these to taste.

## Command-line options

`install.bat` can configure the most common flags. You can also edit `start-webui.bat` or `webui-user.bat` directly.

| Flag | Purpose |
| --- | --- |
| `--no-download-sd-model` | Do not download a default checkpoint |
| `--theme dark` | Start with the dark theme |
| `--uv` | Use the `uv` package manager |
| `--uv-symlink` | Use `uv` with symlinked packages to save disk space |
| `--flash` | Install FlashAttention |
| `--sage` | Install SageAttention |
| `--cuda-malloc` | Use CUDA malloc for memory allocation |
| `--cuda-stream` | Overlap compute and data transfer |
| `--pin-shared-memory` | Use pinned shared memory |
| `--model-ref PATH` | Use a central models folder |
| `--persistent-patches` | Reuse LoRA patches between generations |
| `--adv-samplers` | Show advanced sampler parameters |
| `--adv-xyz` | Show advanced X/Y/Z Plot options |

Memory-related flags can cause instability or out-of-memory errors on some systems. Disable them if you run into trouble.

## Updating

Run:

```bat
update.bat
```

This performs `git reset --hard` followed by `git pull`. **It discards tracked local modifications.** If you have edited the code, commit or back up your changes first, or update manually:

```bash
git pull
```

## Linux

Windows is the primary target. The included shell scripts can be used on Linux after making them executable:

```bash
chmod +x webui.sh webui-user.sh
./webui-user.sh
```

`webui-user.sh` currently enables `--uv`, so `uv` must be installed.

## Troubleshooting

- **No model loaded** — place an SD1/SDXL checkpoint or an Anima model in `models/diffusion_models/` and refresh the checkpoint list.
- **Anima VAE not found** — check for `models/VAE/qwen_image_vae.safetensors`.
- **Anima text encoder not found** — check for `models/text_encoders/qwen_3_06b_base.safetensors`.
- **Anima Hires. fix runs out of VAE memory** — Sakuga automatically retries with tiled VAE encoding. If it still runs out of memory, lower **Settings → VAE → Tile Size for Tiled VAE** (for example, to `256`) or enable the **Never OOM** VAE option.
- **Tokenizer errors** — make sure `modules_forge/text_processing/tokenizer/` and `tokenizer_2/` have not been deleted. Restore them with `git checkout -- modules_forge/text_processing`.
- **Installer asks about renaming `webui-user.bat`** — accept if you want the installer to generate a clean `start-webui.bat`; the original `webui-user.bat` is kept in Git history if you need it back.

## Credits

Sakuga builds on the work of:

- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and contributors
- [lllyasviel](https://github.com/lllyasviel/stable-diffusion-webui-forge) for Forge
- [Haoming02](https://github.com/Haoming02/sd-webui-forge-classic) for Forge Classic
- [comfyanonymous](https://github.com/comfyanonymous/ComfyUI) and the `ldm_patched` contributors
- [kijai](https://github.com/kijai) and the other community contributors
- The Qwen team for Qwen3
- Google for the T5 tokenizer
- The NVIDIA Cosmos authors and the Anima model authors for the underlying architecture and model

Sakuga-specific changes and the Anima integration are maintained by [otacoo](https://github.com/otacoo).

## License

Sakuga is released under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE) for the full text.

Model weights, tokenizers, and other third-party components remain subject to their own licenses and terms of use.
