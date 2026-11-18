"""Model folder paths with fallbacks for older A1111 and ComfyUI layouts."""

import os

from modules.paths import models_path


def relative_to_any(path, roots):
    abspath = os.path.abspath(path)
    for root in roots:
        root = os.path.abspath(root)
        if abspath == root or abspath.startswith(root + os.sep):
            return os.path.relpath(abspath, root)
    return None


def _dirs(candidates, *, always=()):
    result = []
    seen = set()

    for path in [*always, *candidates]:
        if not path:
            continue
        path = os.path.abspath(path)
        key = os.path.normcase(path)
        if key in seen:
            continue
        if path in always or os.path.isdir(path):
            seen.add(key)
            result.append(path)

    return result


def checkpoint_dirs():
    primary = os.path.join(models_path, "diffusion_models")
    return _dirs(
        [
            primary,
            os.path.join(models_path, "Stable-diffusion"),
            os.path.join(models_path, "checkpoints"),
        ],
        always=[primary],
    )


def lora_dirs(primary=None):
    return _dirs(
        [
            primary,
            os.path.join(models_path, "loras"),
            os.path.join(models_path, "Lora"),
        ],
        always=[primary] if primary else [],
    )


def vae_dirs():
    primary = os.path.join(models_path, "VAE")
    return _dirs([primary, os.path.join(models_path, "vae")], always=[primary])


def text_encoder_dirs():
    primary = os.path.join(models_path, "text_encoders")
    return _dirs([primary, os.path.join(models_path, "text_encoder")], always=[primary])
