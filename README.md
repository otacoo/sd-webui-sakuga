<h1 align="center">Stable Diffusion WebUI Sakuga</h1>

<p align="center">
  <b>A personal, experimental, fork of <a href="https://github.com/Haoming02/sd-webui-forge-classic">Haoming02's Forge Classic</a> with native Anima support.</b>
</p>

## What is Sakuga?

Sakuga is a focused Stable Diffusion WebUI forked from **Haoming02's Forge Classic**. It keeps the lightweight SD1/SDXL workflow of Classic and adds native support for **Anima**.

Current architectures: **SD1**, **SDXL**, and **Anima**. Other model families are not supported.

It is intended for users who want a compact WebUI, fast startup, and Anima support without the extra model families and bloat of larger forks.

## Features

### Anima

- Native Anima inference
- Qwen3-0.6B and T5 text processing, including prompt emphasis and `BREAK`
- `ER SDE` sampler for flow-based models
- 16-channel latent previews
- ControlNet-LLLite adapter for Anima *(experimental)*
- Anima Edit support: reference conditioning through img2img and the native ImageStitch panel *(requires an Anima Edit LoRA)*
- Anima RGBA support: 4-channel VAE decode, transparent PNG saving, and native alpha-channel generation *(requires an Anima RGBA LoRA)*
- Text encoder selection and VAE compatibility fallback

### Forge Classic foundation

- SD1 and SDXL checkpoints
- LoRA, ControlNet, IP-Adapter, MultiDiffusion, soft inpainting, and the usual Classic extensions
- SageAttention, FlashAttention, `fp8`, persistent LoRA patching, and in-place LoRA patching
- Sampler and scheduler selection, including `ER SDE`
- Hires. fix CFG, MaHiRo, RescaleCFG, VAE tiling, and the other Classic optimizations

### Sakuga UI and tooling

- Optional **Hide Styles** setting
- Native **Enqueue** button for queueing prompts during generation, with placement and notification settings
- Native **List all models** / **Unload Models** footer links
- Native **Transparent Background** postprocessing in Extras (Lucida; optional Inspyrenet Rembg)
- Drag-over top tabs to open a tab while dragging files
- Live previews in the full-page image viewer
- Windows `install.bat`, `start-webui.bat`, and `update.bat` helpers

## Requirements

- **Windows 10/11** — primary target
- **Git**
- **Python 3.11.x** — 3.11.9 recommended, or use `uv`
  - On Windows, `webui.bat` automatically looks for Python 3.11 even if another Python version comes first on `PATH`. If yours is in a non-standard location, set `PYTHON` in `webui-user.bat`.
- An NVIDIA GPU with current CUDA drivers is strongly recommended
- Enough disk space for Python packages and model weights
- Optional: `python -m pip install transparent-background` to enable Inspyrenet Rembg in the Transparent Background postprocessor. Lucida works without it.

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

Sakuga does not download models automatically. The WebUI starts without a checkpoint and waits for you to add one.

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

Anima is a 2B model, so 30-50 steps at 896x1152 takes time, and Hires. fix plus a 4x upscaler adds considerably more. For quick iteration, use **Anima-Turbo** at **CFG 1** with **8-12 steps**; use Anima-Aesthetic for final images.

#### Anima Edit

1. Put an Anima Edit LoRA in `models/loras/Anima/` and load it with a normal `<lora:...>` tag.
2. Enable **Settings → Anima → [Anima] Enable Reference**.
3. Use **img2img** with the reference image, or use the **ImageStitch Integrated** panel in txt2img/img2img to provide one or more references.
4. Include the Edit LoRA's trigger words in the prompt.

Anima Edit is reference conditioning, not a normal img2img denoise. Select the Edit LoRA trained for the effect you want.

#### Anima RGBA

1. Put `anima_rgba_vae.safetensors` in `models/vae/` and `anima_rgba_lora.safetensors` in `models/loras/Anima/`.
2. Select the RGBA VAE in **Settings → VAE**, and load the RGBA LoRA at strength `1.0`.
3. Add `transparent background` (or `isolated on a transparent background`) to the prompt.
4. Save as PNG to keep the alpha channel. If the background stays opaque, raise the trigger weight (for example `(transparent background:1.3)`) rather than the LoRA strength.

#### Anima prompting

- Use Danbooru-style tags, natural-language captions, or a mix of both.
- Prefix artist tags with `@`, for example `@club8` or `@takeda hiromitsu`. Without the `@`, artist tags have very little effect.
- Place artist tags near the beginning of the prompt, after the quality/safety tags.
- For **Anima-Aesthetic**, use CFG `3-5` and avoid `score_*` tags; the aesthetic version already produces high-quality images.
- For **Anima-Turbo**, use CFG `1` and `8-12` steps.
- Live previews during sampling are approximate; the final image is decoded through the full VAE.

## Command-line options

`install.bat` can configure the most common flags. You can also edit `start-webui.bat` or `webui-user.bat` directly.

| Flag | Purpose |
| --- | --- |
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
- **Anima-Base produces blobs or very soft images** — Base is the unrefined model and needs a normal 30-50 step run with CFG 4-6, quality tags, and an anime/illustration-style prompt. It is not intended for photorealism. Test at 1024x1024 without Hires. fix first. Anima now selects BF16 automatically on supported GPUs; if you force FP16 with `--unet-in-fp16` or use an older GPU, Base may collapse to a constant latent. Use `--unet-in-bf16` or `--no-half` to rule that out.
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
