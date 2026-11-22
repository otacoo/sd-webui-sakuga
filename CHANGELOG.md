# Changelog

All notable changes to Sakuga are listed here.

## Unreleased

### New options & changes

- **New feature:** Native Anima support, including text processing, `ER SDE`, latent previews, LoRA loading, and experimental ControlNet-LLLite.
- **New feature:** Anima Edit reference conditioning via img2img and ImageStitch. Requires an Anima Edit LoRA and `[Anima] Enable Reference`; reload the UI after changing it.
- **New feature:** Anima RGBA support with 4-channel VAE decoding, transparent PNG saving, and alpha generation. Requires an RGBA VAE and LoRA.
- **New option:** `[Anima] Enable Shift` adds the Shift slider for Anima models. Higher values add variation.
- **New feature:** Native PyTorch 2.15 with SDPA; xformers is no longer recommended.
- **New option:** Add support for Comfy-Kitchen INT8 attention for Anima on supported NVIDIA GPUs
- **New feature:** Queue the current prompt while another generation runs, with placement and notification settings.
- **New feature:** Footer links to list all models or unload loaded models; can be hidden.
- **New feature:** Transparent Background postprocessing in Extras, with optional background-removal engines.
- **New feature:** Drag a file over a tab to switch tabs while dragging.
- **New feature:** Delete the selected gallery image from the image viewer.
- **New feature:** Extra Networks folder filters with an `All` button; the last folder is remembered per page.
- **New option:** Hidden Subdirectories for Extra Networks, with comma-separated names or relative paths.
- **New option:** Choose whether LoRA prompts insert at the cursor or the end. Default: end of prompt.
- **New feature:** Extra Networks metadata shows the model path, and the card action can open it in Explorer.
- **New feature:** Prompt boxes grow and shrink automatically; positive prompts keep a taller minimum height.
- **New feature:** Width and Height sliders go up to 4096 pixels (16 MP), including Hires. fix fields; old `ui-config.json` limits update automatically.
- **Change:** LoRA cards are always shown. Version-based hiding was removed, so LyCORIS and other working adapters no longer disappear.
- **Change:** Legacy model folders remain as fallbacks behind the ComfyUI-style folders.
- **Change:** The compact prompt layout is the only layout; face restoration was removed.
- **Change:** Sakuga no longer downloads a default checkpoint; it can start without a model. `--no-download-sd-model` remains a hidden no-op.
- **Change:** Startup uses the compact banner and styled rows; generation and model-loading messages keep the Classic format. Startup checks are lighter.
- **Change:** `config.json` is created by the installer for new installs, not every startup.
- **Change:** Less background work in Extra Networks, tabs, galleries, inpaint resizing, and Enqueue/gamepad polling. Large model libraries feel more responsive.

### Bug fixes & others

- Extra Networks search and sorting only update the current page, so large libraries load faster.
- Editing card metadata no longer rebuilds every Extra Networks page.
- Extra Networks previews stay lazy-loaded when card order does not change.
- Fixed ADetailer masks for Anima latent models.
- Fixed missing gallery images with junction or symlink output folders.
- Fixed masked inpaint on some models and reduced Anima Hires. fix memory use.
- Fixed Anima Hires. fix failures at some scales by aligning to the 8-pixel VAE stride.
- Reduced unnecessary startup warnings and checks.
