import os
import json
import torch
import safetensors.torch

from modules import paths


MODELS_DIR = os.path.join(paths.models_path)
VAE_DIR = os.path.join(MODELS_DIR, "VAE")
TE_DIR = os.path.join(MODELS_DIR, "text_encoder")

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_processing", "tokenizer")
TOKENIZER2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_processing", "tokenizer_2")

ANIMA_VAE_FILENAME = "qwen_image_vae.safetensors"
ANIMA_TE_FILENAME = "qwen_3_06b_base.safetensors"


def _find_file_in(dir_path, filename):
    if not os.path.isdir(dir_path):
        return None
    for f in os.listdir(dir_path):
        if f.lower() == filename.lower():
            return os.path.join(dir_path, f)
    return None


def find_anima_vae():
    return _find_file_in(VAE_DIR, ANIMA_VAE_FILENAME)


def find_anima_text_encoder():
    return _find_file_in(TE_DIR, ANIMA_TE_FILENAME)


def is_anima_model(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    index_path = os.path.join(path, "model_index.json")
    if not os.path.isfile(index_path):
        return False
    try:
        with open(index_path, "r") as f:
            data = json.load(f)
        return data.get("_class_name") == "AnimaPipeline"
    except (json.JSONDecodeError, KeyError):
        return False


def strip_state_dict_prefix(sd):
    prefixes = ("model.diffusion_model.", "net.")
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in sd):
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return sd


def process_anima(transformer_sd, text_encoder_sd):
    keys = list(transformer_sd.keys())
    for k in keys:
        if k.startswith("llm_adapter"):
            text_encoder_sd[k] = transformer_sd.pop(k)


def detect_anima_config_from_state_dict(state_dict):
    sd = strip_state_dict_prefix(state_dict)
    config = {}
    config["in_channels"] = 16
    config["out_channels"] = 16
    config["patch_spatial"] = 2
    config["patch_temporal"] = 1
    config["model_channels"] = 2048
    config["num_blocks"] = 28
    config["num_heads"] = 16
    config["concat_padding_mask"] = True
    config["crossattn_emb_channels"] = 1024

    if "x_embedder.proj.1.weight" in sd:
        embed_weight = sd["x_embedder.proj.1.weight"]
        config["in_channels"] = embed_weight.shape[1] // 4 - int(config["concat_padding_mask"])
        config["model_channels"] = embed_weight.shape[0]

    if "blocks.0.mlp.layer1.weight" in sd:
        num_blocks = 0
        while f"blocks.{num_blocks}.mlp.layer1.weight" in sd:
            num_blocks += 1
        config["num_blocks"] = num_blocks

    if "blocks.0.cross_attn.k_proj.weight" in sd:
        k_proj = sd["blocks.0.cross_attn.k_proj.weight"]
        config["crossattn_emb_channels"] = k_proj.shape[1]

    return config


class AnimaModelConfig:
    sampling_settings = {"multiplier": 1.0, "shift": 3.0}
    memory_usage_factor = 1.32


def _load_safetensors_sd(filepath):
    from ldm_patched.modules import model_management
    sd = {}
    with safetensors.torch.safe_open(filepath, framework="pt") as sf:
        for k in sf.keys():
            sd[k] = sf.get_tensor(k)
    return sd


def load_state_dict_strict(model, sd, log_name=None, ignore_start=None):
    log_name = log_name or type(model).__name__
    model_sd = model.state_dict()
    missing = list(model_sd.keys() - sd.keys())
    unexpected = list(sd.keys() - model_sd.keys())
    loaded = 0
    mismatched = 0
    for k in sd:
        if k in model_sd and sd[k].shape == model_sd[k].shape:
            model_sd[k].copy_(sd[k])
            loaded += 1
        elif k in model_sd:
            print(f"  {log_name} size mismatch {k}: checkpoint {tuple(sd[k].shape)} vs model {tuple(model_sd[k].shape)}")
            mismatched += 1
    if isinstance(ignore_start, str):
        missing = [x for x in missing if not x.startswith(ignore_start)]
        unexpected = [x for x in unexpected if not x.startswith(ignore_start)]
    elif ignore_start:
        missing = [x for x in missing if x not in ignore_start]
        unexpected = [x for x in unexpected if x not in ignore_start]
    print(f"  {log_name}: loaded {loaded} keys, {mismatched} mismatched, {len(missing)} missing, {len(unexpected)} unexpected")
    if mismatched > 0:
        print(f"    !! SIZE MISMATCHES in {log_name}")
    if len(unexpected) > 0:
        print(f"    !! UNEXPECTED keys in {log_name}: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def _convert_vae_sd_5d_to_4d(sd):
    out = {}
    skipped = 0
    for k, v in sd.items():
        if "time_conv" in k:
            skipped += 1
            continue
        if v.dim() == 5 and v.shape[2] in (1, 3):
            out[k] = v[:, :, -1, :, :].contiguous()
        else:
            out[k] = v
    if skipped:
        print(f"  Skipped {skipped} temporal keys (not needed for image-only)")
    return out


def load_anima_vae(filepath=None):
    from modules_forge.nn.wan_vae_2d import Qwen2DVAE
    from modules_forge.nn._vae import Wan21

    if filepath is None:
        filepath = find_anima_vae()
    if filepath is None or not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"{ANIMA_VAE_FILENAME} not found in {VAE_DIR}. "
            f"Please place the Anima VAE weights file in the VAE models folder."
        )

    print(f"Loading Anima VAE: {filepath}")
    vae_sd = _load_safetensors_sd(filepath)

    has_5d = any(v.dim() == 5 for v in vae_sd.values())
    if has_5d:
        print("  Converting 5D (temporal) convolution weights to 4D")
        vae_sd = _convert_vae_sd_5d_to_4d(vae_sd)

    # Convert 4D gamma tensors to 3D to match model shape
    for k, v in vae_sd.items():
        if k.endswith('.gamma') and v.dim() == 4:
            vae_sd[k] = v.squeeze(-1)

    vae = Qwen2DVAE.from_config({})
    load_state_dict_strict(vae, vae_sd, log_name="Anima VAE")
    vae.latent_format = Wan21()
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def load_anima_text_encoder(filepath=None):
    from modules_forge.nn.llama import Qwen3_06B

    if filepath is None:
        filepath = find_anima_text_encoder()
    if filepath is None or not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"{ANIMA_TE_FILENAME} not found in {TE_DIR}. "
            f"Please place the Qwen3-0.6B text encoder weights file in the text_encoder models folder."
        )

    print(f"Loading Anima Text Encoder: {filepath}")
    te_sd = _load_safetensors_sd(filepath)

    text_encoder = Qwen3_06B({})
    load_state_dict_strict(text_encoder, te_sd, log_name="Qwen3_06B", ignore_start="lm_head.")
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    return text_encoder


def load_anima_transformer(state_dict):
    from modules_forge.nn.anima import Anima as CosmosAnima

    sd = strip_state_dict_prefix(state_dict)
    config = detect_anima_config_from_state_dict(state_dict)

    print(f"Building Anima transformer: {config['model_channels']}ch, {config['num_blocks']} blocks, {config['crossattn_emb_channels']} crossattn")
    transformer = CosmosAnima(
        max_img_h=1024,
        max_img_w=1024,
        max_frames=1,
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        patch_spatial=config["patch_spatial"],
        patch_temporal=config["patch_temporal"],
        model_channels=config["model_channels"],
        concat_padding_mask=config["concat_padding_mask"],
        crossattn_emb_channels=config["crossattn_emb_channels"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        rope_h_extrapolation_ratio=4.0,
        rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0,
        pos_emb_cls="rope3d",
        use_adaln_lora=True,
        adaln_lora_dim=256,
    )

    load_state_dict_strict(transformer, sd, log_name="Anima Transformer")
    transformer = transformer.half()
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False

    transformer.storage_dtype = torch.float16
    transformer.computation_dtype = torch.float16
    transformer.load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.initial_device = torch.device("cpu")
    transformer.offload_device = torch.device("cpu")

    return transformer


def load_anima_tokenizers():
    from transformers import Qwen2Tokenizer, T5TokenizerFast

    tokenizer = None
    tokenizer_2 = None

    if os.path.isdir(TOKENIZER_DIR) and os.path.isfile(os.path.join(TOKENIZER_DIR, "tokenizer_config.json")):
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(TOKENIZER_DIR)
        except Exception as e:
            print(f"Failed to load Qwen2 tokenizer from {TOKENIZER_DIR}: {e}")

    if tokenizer is None:
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        except Exception as e:
            print(f"Failed to load Qwen2 tokenizer from HuggingFace cache: {e}")

    if tokenizer is None:
        raise FileNotFoundError(
            f"Qwen2 tokenizer not found. Please place tokenizer files in:\n  {TOKENIZER_DIR}/"
        )

    if os.path.isdir(TOKENIZER2_DIR) and os.path.isfile(os.path.join(TOKENIZER2_DIR, "tokenizer_config.json")):
        try:
            tokenizer_2 = T5TokenizerFast.from_pretrained(TOKENIZER2_DIR)
        except Exception as e:
            print(f"Failed to load T5 tokenizer from {TOKENIZER2_DIR}: {e}")

    if tokenizer_2 is None:
        try:
            tokenizer_2 = T5TokenizerFast.from_pretrained("google-t5/t5-base")
        except Exception as e:
            print(f"Failed to load T5 tokenizer from HuggingFace cache: {e}")

    if tokenizer_2 is None:
        raise FileNotFoundError(
            f"T5 tokenizer not found. Please place tokenizer files in:\n  {TOKENIZER2_DIR}/"
        )

    return tokenizer, tokenizer_2


def load_anima_components(path):
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Anima model directory not found: {path}")

    index_path = os.path.join(path, "model_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"model_index.json not found in {path}")

    with open(index_path, "r") as f:
        index = json.load(f)

    if index.get("_class_name") != "AnimaPipeline":
        raise ValueError(f"Not an Anima model: {path}")

    te_path = os.path.join(path, "text_encoder")
    vae_path = os.path.join(path, "vae")

    transformer_sd = {}
    transformer_dir = os.path.join(path, "transformer")
    transformer_file = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
    if not os.path.isfile(transformer_file):
        raise FileNotFoundError(f"Transformer safetensors not found: {transformer_file}")
    transformer_sd = _load_safetensors_sd(transformer_file)

    te_sd = {}
    te_file = os.path.join(te_path, "diffusion_pytorch_model.safetensors")
    if os.path.isfile(te_file):
        te_sd = _load_safetensors_sd(te_file)

    process_anima(transformer_sd, te_sd)

    transformer = load_anima_transformer(transformer_sd)
    vae = load_anima_vae(os.path.join(vae_path, "diffusion_pytorch_model.safetensors"))

    if te_sd:
        text_encoder = load_anima_text_encoder(te_file)
        load_state_dict_strict(text_encoder, te_sd, log_name="Qwen3_06B (llm_adapter)", ignore_start="lm_head.")
    else:
        text_encoder = load_anima_text_encoder(te_file)

    tokenizer, tokenizer_2 = load_anima_tokenizers()

    for model in [transformer, vae, text_encoder]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return {
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "vae": vae,
        "transformer": transformer,
    }


def load_anima_from_state_dict(state_dict, vae_path=None, te_path=None):
    transformer_sd = strip_state_dict_prefix(state_dict)

    te_sd = {}
    process_anima(transformer_sd, te_sd)

    transformer = load_anima_transformer(state_dict)
    vae = load_anima_vae(vae_path)

    if te_sd:
        text_encoder = load_anima_text_encoder(te_path)
        load_state_dict_strict(text_encoder, te_sd, log_name="Qwen3_06B (llm_adapter)", ignore_start="lm_head.")
    else:
        text_encoder = load_anima_text_encoder(te_path)

    tokenizer, tokenizer_2 = load_anima_tokenizers()

    for model in [transformer, vae, text_encoder]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return {
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "vae": vae,
        "transformer": transformer,
    }


def detect_anima_from_state_dict(state_dict: dict) -> bool:
    anima_markers = ("blocks.0.mlp.layer1.weight", "x_embedder.proj.1.weight",
                     "llm_adapter.blocks.0.cross_attn.q_proj.weight")
    for k in state_dict:
        for marker in anima_markers:
            if k.endswith(marker):
                return True
    return False


def is_anima_state_dict(state_dict: dict) -> bool:
    return detect_anima_from_state_dict(state_dict)