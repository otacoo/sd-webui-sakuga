import torch

from ldm_patched.modules import model_management
from modules_forge.forge_loader import ForgeObjects
from modules_forge.text_processing.anima_engine import AnimaTextProcessingEngine


class VAEWrapper:
    def __init__(self, model=None, device=None, dtype=None):
        self.latent_channels = model.z_dim
        self.downscale_ratio = 8
        self.upscale_ratio = 8

        self.first_stage_model = model.eval()

        self.device = device or model_management.vae_device()
        offload_device = model_management.vae_offload_device()
        self.vae_dtype = dtype or model_management.vae_dtype()
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = model_management.intermediate_device()

        self.patcher = model_management.ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device,
        )

        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype)

    @staticmethod
    def process_input(image):
        return image * 2.0 - 1.0

    @staticmethod
    def process_output(image):
        return image.add(1.0).div(2.0).clamp(0.0, 1.0)

    def encode(self, pixel_samples):
        if pixel_samples.ndim != 4:
            raise ValueError(f"VAE encode expects 4D input (B, H, W, C), got {pixel_samples.shape}")
        _samples = pixel_samples.movedim(-1, 1)
        memory_used = self.memory_used_encode(_samples.shape, self.vae_dtype)
        model_management.load_models_gpu([self.patcher], memory_required=memory_used)
        pixels_in = self.process_input(_samples).to(self.vae_dtype).to(self.device)
        out = self.first_stage_model.encode(pixels_in).to(self.output_device).float()
        return out

    def decode(self, samples):
        if samples.ndim != 4:
            raise ValueError(f"VAE decode expects 4D input (B, C, H, W), got {samples.shape}")
        return self.decode_tiled_(samples)

    def decode_tiled_(self, samples, tile_x=0, tile_y=0, overlap=0):
        from ldm_patched.modules.utils import tiled_scale, ProgressBar, get_tiled_scale_steps

        tile_x = int(tile_x or 32)
        tile_y = int(tile_y or 32)
        overlap = int(overlap or 4)

        steps = samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        pbar = ProgressBar(steps, title="VAE tiled decode")

        model_management.load_models_gpu([self.patcher])

        def decode_fn(a):
            return (self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)) + 1.0).float()

        output = torch.clamp(
            (
                tiled_scale(
                    samples, decode_fn,
                    tile_x // 2, tile_y * 2, overlap,
                    upscale_amount=self.downscale_ratio,
                    output_device=self.output_device, pbar=pbar,
                )
                + tiled_scale(
                    samples, decode_fn,
                    tile_x * 2, tile_y // 2, overlap,
                    upscale_amount=self.downscale_ratio,
                    output_device=self.output_device, pbar=pbar,
                )
                + tiled_scale(
                    samples, decode_fn,
                    tile_x, tile_y, overlap,
                    upscale_amount=self.downscale_ratio,
                    output_device=self.output_device, pbar=pbar,
                )
            ) / 6.0,
            min=0.0, max=1.0,
        )
        return output.movedim(1, -1)


class AnimaEngine:
    def __init__(self, estimated_config, huggingface_components):
        self.is_inpaint = False
        self.model_config = estimated_config

        text_encoder = huggingface_components["text_encoder"]
        qwen_tokenizer = huggingface_components["tokenizer"]
        t5_tokenizer = huggingface_components["tokenizer_2"]
        vae_model = huggingface_components["vae"]
        transformer = huggingface_components["transformer"]

        from modules_forge.k_prediction import PredictionDiscreteFlow
        from modules_forge.anima_model_adapter import KModel
        from modules_forge.unet_patcher import UnetPatcher

        k_predictor = PredictionDiscreteFlow(self.model_config)

        k_model = KModel(
            model=transformer,
            config=self.model_config,
            k_predictor=k_predictor,
        )

        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        unet = UnetPatcher(
            k_model,
            load_device=load_device,
            offload_device=offload_device,
            current_device=offload_device,
            weight_inplace_update=False,
        )

        self._clip_patcher = model_management.ModelPatcher(
            model=text_encoder,
            load_device=model_management.text_encoder_device(),
            offload_device=model_management.text_encoder_offload_device(),
        )

        vae = VAEWrapper(model=vae_model, dtype=torch.float16)

        class ClipWrapper:
            def __init__(self, patcher, te, tok1, tok2):
                self.patcher = patcher
                self.cond_stage_model = te
                class TokDict:
                    def __init__(self, q, t):
                        self.qwen3_06b = q
                        self.t5xxl = t
                self.tokenizer = TokDict(tok1, tok2)

        clip_wrapper = ClipWrapper(self._clip_patcher, text_encoder, qwen_tokenizer, t5_tokenizer)

        self.text_processing_engine_anima = AnimaTextProcessingEngine(
            text_encoder=text_encoder,
            qwen_tokenizer=qwen_tokenizer,
            t5_tokenizer=t5_tokenizer,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip_wrapper, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True
        self.use_shift = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        model_management.load_model_gpu(self._clip_patcher)
        return self.text_processing_engine_anima(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_anima.tokenize([prompt])[0][0])
        return token_count, max(512, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        samples = []
        batch = x.size(0)
        for b in range(batch):
            y = x[b].unsqueeze(0)
            sample = self.forge_objects.vae.encode(y.movedim(1, -1) * 0.5 + 0.5)
            sample = self.forge_objects.vae.first_stage_model.process_in(sample)
            samples.append(sample)
        return torch.cat(samples).to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        samples = []
        batch = x.size(0)
        for b in range(batch):
            y = x[b].unsqueeze(0)
            sample = self.forge_objects.vae.first_stage_model.process_out(y)
            sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
            samples.append(sample)
        return torch.cat(samples).to(x)
