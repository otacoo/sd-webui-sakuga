import dataclasses
from typing import Callable

import k_diffusion
import numpy as np
import torch
from modules import shared
from scipy import stats


def to_d(x: torch.Tensor, sigma: float, denoised: torch.Tensor):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / sigma


k_diffusion.sampling.to_d = to_d


@dataclasses.dataclass
class Scheduler:
    name: str
    label: str
    function: Callable

    default_rho: float = -1.0
    need_inner_model: bool = False
    aliases: list[str] = None


def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, n)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(inner_model.t_to_sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = len(inner_model.sigmas) / n
    for x in range(n):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def uniform(n, sigma_min, sigma_max, inner_model, device):
    return inner_model.get_sigmas(n).to(device)


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))
    sigs = [inner_model.t_to_sigma(ts) for ts in torch.linspace(start, end, n + 1)[:-1]]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def _flow_sigma_table(inner_model):
    predictor = getattr(inner_model, "predictor", None)
    if predictor is None or getattr(predictor, "prediction_type", None) != "const":
        return None
    return predictor.sigmas


def _sigma_to_flow_u(sigma, table):
    """Map a flow sigma to the predictor's normalized [0, 1] table position."""
    table = table.float()
    log_sigma = torch.log(torch.as_tensor(sigma, dtype=table.dtype, device=table.device))
    log_table = table.log()
    idx = torch.searchsorted(log_table, log_sigma).clamp(1, len(table) - 1)
    lo, hi = log_table[idx - 1], log_table[idx]
    w = ((log_sigma - lo) / (hi - lo)).clamp(0.0, 1.0)
    u = ((idx - 1).float() + w) / (len(table) - 1)
    return float(u.clamp(0.0, 1.0))


def _flow_u_to_sigmas(u_sigmas, table):
    """Map normalized flow positions back to sigmas."""
    table = table.float().to(u_sigmas.device)
    u = u_sigmas.clamp(0.0, 1.0)
    pos = u * (len(table) - 1)
    lo = pos.floor().long().clamp(0, len(table) - 1)
    hi = (lo + 1).clamp(max=len(table) - 1)
    w = pos - lo
    log_table = table.log()
    out = torch.exp((1 - w) * log_table[lo] + w * log_table[hi])
    out[-1] = 0.0
    return out


def flow_shifted_scheduler(func, n, sigma_min, sigma_max, device, inner_model=None, **kwargs):
    """Apply sigma schedules in the flow model's shifted timestep domain.

    Karras/Exponential schedules are defined over the noise level.  For flow
    models the predictor's sigma table already contains the shift, so running
    those schedules directly in sigma space leaves too few steps at high noise
    and produces blank or noisy images.  Run the schedule over the normalized
    table index and map the result back to sigmas instead.
    """
    table = _flow_sigma_table(inner_model) if inner_model is not None else None
    if table is None:
        return func(n, sigma_min, sigma_max, device=device, **kwargs)

    u_min = _sigma_to_flow_u(sigma_min, table)
    u_max = _sigma_to_flow_u(sigma_max, table)
    u_sigmas = func(n, u_min, u_max, device=device, **kwargs)
    return _flow_u_to_sigmas(u_sigmas, table)


def get_flow_sigmas_karras(n, sigma_min, sigma_max, device="cpu", inner_model=None, rho=7.0):
    return flow_shifted_scheduler(k_diffusion.sampling.get_sigmas_karras, n, sigma_min, sigma_max, device, inner_model=inner_model, rho=rho)


def get_flow_sigmas_exponential(n, sigma_min, sigma_max, device="cpu", inner_model=None):
    return flow_shifted_scheduler(k_diffusion.sampling.get_sigmas_exponential, n, sigma_min, sigma_max, device, inner_model=inner_model)


def get_flow_sigmas_polyexponential(n, sigma_min, sigma_max, device="cpu", inner_model=None, rho=1.0):
    return flow_shifted_scheduler(k_diffusion.sampling.get_sigmas_polyexponential, n, sigma_min, sigma_max, device, inner_model=inner_model, rho=rho)


def _loglinear_interp(t_steps, num_steps):
    """Performs log-linear interpolation of a given array of decreasing numbers"""
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device):
    """https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html"""

    if getattr(shared.sd_model, "parameterization", None) == "flow":
        # SD AYS values are scaled by the model's actual sigma_max so they
        # stay in the flow model's [0, 1] sigma range.
        sigmas = [sigma_max, sigma_max / 2.257, sigma_max / 3.785, sigma_max / 5.418, sigma_max / 7.749, sigma_max / 10.469, sigma_max / 15.176, sigma_max / 22.415, sigma_max / 36.629, sigma_max / 96.151, 0.002]
    elif shared.sd_model.is_sdxl:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    else:
        sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

    if n != len(sigmas):
        sigmas = np.append(_loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)


def get_align_your_steps_sigmas_GITS(n, sigma_min, sigma_max, device):
    if getattr(shared.sd_model, "parameterization", None) == "flow":
        sigmas = [sigma_max, sigma_max / 3.087, sigma_max / 5.693, sigma_max / 9.558, sigma_max / 14.81, sigma_max / 22.42, sigma_max / 34.96, sigma_max / 54.53, sigma_max / 81.65, sigma_max / 115.1, 0.002]
    elif shared.sd_model.is_sdxl:
        sigmas = [14.615, 4.734, 2.567, 1.529, 0.987, 0.652, 0.418, 0.268, 0.179, 0.127, 0.029]
    else:
        sigmas = [14.615, 4.617, 2.507, 1.236, 0.702, 0.402, 0.240, 0.156, 0.104, 0.094, 0.029]

    if n != len(sigmas):
        sigmas = np.append(_loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)


def linear_quadratic(n, sigma_min, sigma_max, device, *, threshold_noise=0.025):
    if n == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        linear_steps = n // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * n
        quadratic_steps = n - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
        const = quadratic_coef * (linear_steps**2)
        quadratic_sigma_schedule = [quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, n)]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule).to(device) * sigma_max


def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    smax = ((2 / np.pi) * np.arctan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / np.pi) * np.arctan(-slope * ((steps - 1) - pivot)) + 1) / 2

    srange = smax - smin
    sscale = start - end

    sigmas = [((((2 / np.pi) * np.arctan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end for x in range(steps)]

    return sigmas


def bong_tangent_scheduler(n, sigma_min, sigma_max, device, *, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False):
    """https://github.com/ClownsharkBatwing/RES4LYF/blob/main/sigmas.py#L4076"""
    n += 2

    midpoint = int((n * pivot_1 + n * pivot_2) / 2)
    pivot_1 = int(n * pivot_1)
    pivot_2 = int(n * pivot_2)

    slope_1 = slope_1 / (n / 40)
    slope_2 = slope_2 / (n / 40)

    stage_2_len = n - midpoint
    stage_1_len = n - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)

    tan_sigmas_1 = tan_sigmas_1[:-1]
    if pad:
        tan_sigmas_2 = tan_sigmas_2 + [0]

    tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

    return tan_sigmas.to(device)


def kl_optimal(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas


def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = max(len(inner_model.sigmas) // n, 1)
    x = 1
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def beta_scheduler(n, sigma_min, sigma_max, inner_model, device):
    """
    Beta scheduler
    Based on "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)
    """
    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta

    total_timesteps = len(inner_model.sigmas) - 1
    ts = 1 - np.linspace(0, 1, n, endpoint=False)
    ts = np.rint(stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(inner_model.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def turbo_scheduler(n, sigma_min, sigma_max, inner_model, device):
    unet = inner_model.inner_model.forge_objects.unet
    timesteps = torch.flip(torch.arange(1, n + 1) * float(1000.0 / n) - 1, (0,)).round().long().clip(0, 999)
    predictor = getattr(unet.model, "predictor", None) or getattr(unet.model, "model_sampling", None)
    if predictor is not None:
        try:
            sigmas = predictor.sigmas[timesteps.to(predictor.sigmas.device)].to(device)
        except (AttributeError, IndexError):
            sigmas = predictor.sigma(timesteps)
    else:
        sigmas = unet.model.model_sampling.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas.to(device)


schedulers = [
    Scheduler("automatic", "Automatic", None),
    Scheduler("karras", "Karras", get_flow_sigmas_karras, default_rho=7.0, need_inner_model=True),
    Scheduler("exponential", "Exponential", get_flow_sigmas_exponential, need_inner_model=True),
    Scheduler("polyexponential", "Polyexponential", get_flow_sigmas_polyexponential, default_rho=1.0, need_inner_model=True),
    Scheduler("normal", "Normal", normal_scheduler, need_inner_model=True),
    Scheduler("simple", "Simple", simple_scheduler, need_inner_model=True),
    Scheduler("uniform", "Uniform", uniform, need_inner_model=True),
    Scheduler("sgm_uniform", "SGM Uniform", sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler("linear_quadratic", "Linear Quadratic", linear_quadratic),
    Scheduler("kl_optimal", "KL Optimal", kl_optimal),
    Scheduler("ddim", "DDIM", ddim_scheduler, need_inner_model=True),
    Scheduler("align_your_steps", "Align Your Steps", get_align_your_steps_sigmas),
    Scheduler("align_your_steps_GITS", "Align Your Steps GITS", get_align_your_steps_sigmas_GITS),
    Scheduler("beta", "Beta", beta_scheduler, need_inner_model=True),
    Scheduler("turbo", "Turbo", turbo_scheduler, need_inner_model=True),
    Scheduler("bong_tangent", "Bong Tangent", bong_tangent_scheduler),
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
