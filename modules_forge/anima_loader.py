import os
import json
import torch
import safetensors.torch

from modules import console, paths


MODELS_DIR = os.path.abspath(paths.models_path)
VAE_DIR = os.path.join(MODELS_DIR, "VAE")
TE_DIR = os.path.join(MODELS_DIR, "text_encoders")

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


def _load_safetensors_sd(filepath):
    sd = {}
    with safetensors.torch.safe_open(filepath, framework="pt") as sf:
        for k in sf.keys():
            sd[k] = sf.get_tensor(k)
    return sd


def load_state_dict_strict(model, sd, log_name=None, ignore_start=None, label=None):
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
            console.warn(f"{log_name} size mismatch {k}: checkpoint {tuple(sd[k].shape)} vs model {tuple(model_sd[k].shape)}")
            mismatched += 1
    if isinstance(ignore_start, str):
        missing = [x for x in missing if not x.startswith(ignore_start)]
        unexpected = [x for x in unexpected if not x.startswith(ignore_start)]
    elif ignore_start:
        missing = [x for x in missing if x not in ignore_start]
        unexpected = [x for x in unexpected if x not in ignore_start]
    summary = f"{loaded} keys loaded"
    if mismatched or missing or unexpected:
        summary += f" ({mismatched} mismatched, {len(missing)} missing, {len(unexpected)} unexpected)"
    console.kv(label or log_name, summary, width=20, indent=4)
    if mismatched > 0:
        console.warn(f"SIZE MISMATCHES in {log_name}")
    if len(unexpected) > 0:
        console.warn(f"UNEXPECTED keys in {log_name}: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def load_anima_vae(filepath=None):
    from modules_forge.nn._vae import Wan21

    if filepath is None:
        filepath = find_anima_vae()
    if filepath is None or not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"{ANIMA_VAE_FILENAME} not found in {VAE_DIR}. "
            f"Please place the Anima VAE weights file in the VAE models folder."
        )

    console.kv("VAE", os.path.basename(filepath), width=20, indent=4)
    vae_sd = _load_safetensors_sd(filepath)

    has_5d = any(v.dim() == 5 for v in vae_sd.values())
    if has_5d:
        from modules_forge.nn.wan_vae import WanVAE

        encoder_conv = vae_sd.get("encoder.conv1.weight")
        decoder_conv = vae_sd.get("decoder.head.2.weight")
        image_channels = encoder_conv.shape[1] if encoder_conv is not None and encoder_conv.dim() == 5 else 3
        conv_out_channels = decoder_conv.shape[0] if decoder_conv is not None and decoder_conv.dim() == 5 else 3

        console.kv("VAE type", f"3D temporal ({image_channels}ch in / {conv_out_channels}ch out)", width=20, indent=4)
        vae = WanVAE(
            base_dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temporal_downsample=[False, True, True],
            image_channels=image_channels,
            conv_out_channels=conv_out_channels,
            dropout=0.0,
        )
        load_state_dict_strict(vae, vae_sd, log_name="Anima VAE", label="VAE weights")
    else:
        from modules_forge.nn.wan_vae_2d import Qwen2DVAE

        for k, v in vae_sd.items():
            if k.endswith('.gamma') and v.dim() == 4:
                vae_sd[k] = v.squeeze(-1)

        console.kv("VAE type", "2D", width=20, indent=4)
        vae = Qwen2DVAE.from_config({})
        load_state_dict_strict(vae, vae_sd, log_name="Anima VAE", label="VAE weights")

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
            f"Please place the Qwen3-0.6B text encoder weights file in the text_encoders models folder."
        )

    console.kv("Text encoder", os.path.basename(filepath), width=20, indent=4)
    te_sd = _load_safetensors_sd(filepath)

    text_encoder = Qwen3_06B({})
    ignore = [k for k in text_encoder.state_dict() if k.startswith(("lm_head.", "llm_adapter."))]
    load_state_dict_strict(text_encoder, te_sd, log_name="Qwen3_06B", label="Text weights", ignore_start=ignore)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    return text_encoder


def load_anima_transformer(state_dict):
    from modules_forge.nn.anima import Anima as CosmosAnima

    sd = strip_state_dict_prefix(state_dict)
    config = detect_anima_config_from_state_dict(sd)

    console.kv("Transformer", f"{config['model_channels']}ch, {config['num_blocks']} blocks, {config['crossattn_emb_channels']} crossattn", width=20, indent=4)
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

    load_state_dict_strict(transformer, sd, log_name="Anima Transformer", label="Transformer weights")

    from ldm_patched.modules import model_management
    model_params = sum(param.numel() for param in transformer.parameters())
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = model_management.unet_dtype(model_params=model_params)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        dtype = torch.bfloat16 if bf16_supported else torch.float16
    force_fp16 = (
        getattr(model_management.args, "unet_in_fp16", False)
        or getattr(model_management.args, "all_in_fp16", False)
        or getattr(model_management, "FORCE_FP16", False)
    )
    if dtype == torch.float16 and bf16_supported and not force_fp16:
        dtype = torch.bfloat16

    transformer = transformer.to(dtype)
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False

    if model_management.ck_enabled():
        from modules_forge.nn.anima import ck_attention_op

        console.kv("Attention", "Comfy-Kitchen INT8", width=20, indent=4)
        for module in transformer.modules():
            if hasattr(module, "attn_op"):
                module.attn_op = ck_attention_op
    else:
        console.kv("Attention", "PyTorch SDPA", width=20, indent=4)

    transformer.storage_dtype = dtype
    transformer.computation_dtype = dtype
    transformer.load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.initial_device = torch.device("cpu")
    transformer.offload_device = torch.device("cpu")

    return transformer


def load_anima_tokenizers():
    from transformers import Qwen2Tokenizer, T5TokenizerFast

    tokenizer = None
    tokenizer_2 = None

    if os.path.isfile(os.path.join(TOKENIZER_DIR, "tokenizer_config.json")):
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(TOKENIZER_DIR)
        except Exception as e:
            console.warn(f"Failed to load Qwen2 tokenizer from {TOKENIZER_DIR}: {e}")

    if tokenizer is None:
        console.hint("Qwen2 tokenizer not found locally; downloading Qwen/Qwen3-0.6B")
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        except Exception as e:
            console.warn(f"Failed to load Qwen2 tokenizer from HuggingFace: {e}")

    if tokenizer is None:
        raise FileNotFoundError(
            f"Qwen2 tokenizer not found. Please place tokenizer files in:\n  {TOKENIZER_DIR}/"
        )

    if os.path.isfile(os.path.join(TOKENIZER2_DIR, "tokenizer_config.json")):
        try:
            tokenizer_2 = T5TokenizerFast.from_pretrained(TOKENIZER2_DIR)
        except Exception as e:
            console.warn(f"Failed to load T5 tokenizer from {TOKENIZER2_DIR}: {e}")

    if tokenizer_2 is None:
        console.hint("T5 tokenizer not found locally; downloading google-t5/t5-base")
        try:
            tokenizer_2 = T5TokenizerFast.from_pretrained("google-t5/t5-base")
        except Exception as e:
            console.warn(f"Failed to load T5 tokenizer from HuggingFace: {e}")

    if tokenizer_2 is None:
        raise FileNotFoundError(
            f"T5 tokenizer not found. Please place tokenizer files in:\n  {TOKENIZER2_DIR}/"
        )

    return tokenizer, tokenizer_2


def _finalize_anima_components(transformer, vae, text_encoder):
    tokenizer, tokenizer_2 = load_anima_tokenizers()

    for model in (transformer, vae, text_encoder):
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


def _load_llm_adapter(text_encoder, te_sd):
    base_keys = [k for k in text_encoder.state_dict() if not k.startswith("llm_adapter.")]
    load_state_dict_strict(text_encoder, te_sd, log_name="Qwen3_06B (llm_adapter)", label="LLM adapter", ignore_start=base_keys)


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

    console.section("Loading Anima model")

    te_path = os.path.join(path, "text_encoder")
    vae_path = os.path.join(path, "vae")

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
        _load_llm_adapter(text_encoder, te_sd)
    else:
        text_encoder = load_anima_text_encoder(te_file)

    return _finalize_anima_components(transformer, vae, text_encoder)


def load_anima_from_state_dict(state_dict, vae_path=None, te_path=None):
    console.section("Loading Anima model")
    transformer_sd = strip_state_dict_prefix(state_dict)

    te_sd = {}
    process_anima(transformer_sd, te_sd)

    transformer = load_anima_transformer(state_dict)
    vae = load_anima_vae(vae_path)

    if te_sd:
        text_encoder = load_anima_text_encoder(te_path)
        _load_llm_adapter(text_encoder, te_sd)
    else:
        text_encoder = load_anima_text_encoder(te_path)

    return _finalize_anima_components(transformer, vae, text_encoder)