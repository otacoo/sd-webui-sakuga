# Changelog

All notable changes to Sakuga are listed here.

## Unreleased

### New options & changes

- **New feature:** Native Anima support, including Anima text processing, `ER SDE`, latent previews, LoRA loading, and the experimental ControlNet-LLLite adapter.
- **New feature:** Anima Edit reference conditioning through img2img and the native ImageStitch panel. Requires an Anima Edit LoRA and `Settings → Anima → [Anima] Enable Reference`; changing the setting requires a UI reload.
- **New feature:** Anima RGBA support, including 4-channel VAE decoding, transparent PNG saving, and alpha-channel generation. Requires an Anima RGBA VAE and LoRA.
- **New option:** `[Anima] Enable Shift` shows the Shift slider for Anima models. Higher values produce more variation; the setting only applies to Anima.
- **New feature:** Native PyTorch 2.15 setup with native SDPA. xformers is no longer recommended.
- **New feature:** Enqueue button for queueing the current prompt while another generation is running, with placement and notification settings.
- **New feature:** Footer links to list all models or unload loaded models, with an option to hide them.
- **New feature:** Transparent Background postprocessing in the Extras tab, with optional background-removal engines.
- **New feature:** Drag a file over a top-level tab to switch to that tab while dragging.
- **New feature:** Delete the selected gallery image directly from the image viewer.
- **New feature:** Extra Networks folder filters, labelled `Folders:`, with an `All` button. The last selected folder is remembered per page.
- **New option:** Hidden Subdirectories in Extra Networks, with comma-separated folder names or relative paths.
- **New feature:** Extra Networks metadata shows the model's absolute path, and the card action can open the model in Explorer.
- **New feature:** Positive and negative prompt boxes grow automatically as you type and shrink again when text is removed. Positive prompts keep a taller minimum height than negative prompts.
- **New feature:** Width and Height sliders now go up to 4096 pixels (4096×4096 = 16 MP), including Hires. fix resize fields. Old `ui-config.json` limits are updated automatically.
- **Change:** Legacy model folders such as `Stable-diffusion`, `Lora`, `VAE`, and `text_encoder` remain available as fallbacks behind the ComfyUI-style folders.
- **Change:** The compact prompt layout is now the only layout; face restoration has been removed.
- **Change:** Sakuga no longer downloads a default checkpoint automatically. The WebUI can start without a model and wait for you to add one; `--no-download-sd-model` remains as a hidden compatibility no-op.
- **Change:** Startup output uses the compact Sakuga banner and styled environment/model rows, while generation and model-loading messages keep the original Classic format. Startup checks are lighter.
- **Change:** `config.json` is created by the installer for new installs instead of being recreated at every startup.
- **Change:** Less background UI work when opening Extra Networks, switching tabs, scrolling model cards, updating galleries, resizing the inpaint canvas, and polling Enqueue/gamepad state. Large model libraries should feel more responsive.

### Bug fixes & others

- Extra Networks search and sorting now only touch the current page, so large LoRA/checkpoint libraries open and filter faster.
- Editing one card's metadata no longer rebuilds every Extra Networks page.
- Extra Networks preview images stay lazy-loaded when the card order does not need to change.
- Fixed ADetailer masks for Anima latent models.
- Fixed missing gallery images when output folders use junctions or symlinks.
- Fixed masked inpaint issues on some models and reduced Anima Hires. fix memory use.
- Reduced unnecessary startup warnings and checks.
