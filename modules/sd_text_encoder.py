import glob
import os

from modules import model_paths, paths, shared

te_path = os.path.abspath(os.path.join(paths.models_path, "text_encoders"))
te_dict = {}


def get_filename(filepath):
    return model_paths.relative_to_any(filepath, model_paths.text_encoder_dirs()) or os.path.basename(filepath)


def refresh_te_list():
    te_dict.clear()

    paths_list = []
    for root in model_paths.text_encoder_dirs():
        paths_list += [
            os.path.join(root, "**/*.ckpt"),
            os.path.join(root, "**/*.pt"),
            os.path.join(root, "**/*.safetensors"),
        ]

    if shared.cmd_opts.ckpt_dir is not None and os.path.isdir(shared.cmd_opts.ckpt_dir):
        paths_list += [
            os.path.join(shared.cmd_opts.ckpt_dir, "**/text_encoders/*.ckpt"),
            os.path.join(shared.cmd_opts.ckpt_dir, "**/text_encoders/*.pt"),
            os.path.join(shared.cmd_opts.ckpt_dir, "**/text_encoders/*.safetensors"),
        ]

    candidates = []
    for path in paths_list:
        candidates += glob.iglob(path, recursive=True)

    for filepath in candidates:
        name = get_filename(filepath)
        te_dict[name] = filepath

    te_dict.update(dict(sorted(te_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))


def find_text_encoder_near_checkpoint(checkpoint_file):
    checkpoint_path = os.path.basename(checkpoint_file).rsplit(".", 1)[0]
    for te_file in te_dict.values():
        if os.path.basename(te_file).startswith(checkpoint_path):
            return te_file
    return None
