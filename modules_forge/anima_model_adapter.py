import math

import torch

from modules_forge.k_prediction import PredictionDiscreteFlow


class KModel(torch.nn.Module):
    def __init__(self, model, config, k_predictor=None):
        super().__init__()

        self.config = config

        self.storage_dtype = getattr(model, 'storage_dtype', torch.float16)
        self.computation_dtype = getattr(model, 'computation_dtype', torch.float16)

        self.diffusion_model = model
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)

        self.predictor = k_predictor or PredictionDiscreteFlow(shift=3.0)

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        xc = xc.to(dtype)
        flow_t = self.predictor.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(xc, flow_t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()

        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape):
        input_shapes = [input_shape]
        area = sum(map(lambda s: s[0] * math.prod(s[2:]), input_shapes))

        from ldm_patched.modules import model_management
        if model_management.xformers_enabled() or model_management.flash_enabled():
            return (area * model_management.dtype_size(self.computation_dtype) * 0.01 * 1.32) * (1024 * 1024)
        else:
            return (area * 0.15 * 1.32) * (1024 * 1024)