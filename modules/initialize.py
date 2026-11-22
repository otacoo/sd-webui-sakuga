import importlib
import logging
import os
import sys
import warnings

from modules import console
from modules.timer import startup_timer


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def shush():
    logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)
    logging.getLogger("xformers").addFilter(lambda record: "triton" not in record.getMessage().lower())
    logging.getLogger("torch.utils.flop_counter").addFilter(lambda record: "triton not found" not in record.getMessage())
    warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
    warnings.filterwarnings(action="ignore", category=FutureWarning, message=r"`torch\.jit\.script` is deprecated.*")


def imports():
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    startup_timer.record("import torch")

    import numpy as np
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'complex_'):
        np.complex_ = np.complex128
    if not hasattr(np, 'obj2sctype'):
        np.obj2sctype = lambda obj: np.dtype(obj).type

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    import gradio  # noqa: F401
    startup_timer.record("import gradio")

    from modules import paths, timer, errors  # noqa: F401
    startup_timer.record("setup paths")

    from modules import shared_init
    shared_init.initialize()
    startup_timer.record("initialize shared")

    from modules import processing, gradio_extensons, ui  # noqa: F401
    startup_timer.record("other imports")


def check_versions():
    from modules.shared_cmd_options import cmd_opts

    if not cmd_opts.skip_version_check:
        from modules import errors
        errors.check_versions()


def initialize():
    from modules import initialize_util
    initialize_util.fix_torch_version()
    initialize_util.fix_asyncio_event_loop_policy()
    initialize_util.validate_tls_options()
    initialize_util.configure_sigint_handler()
    initialize_util.configure_opts_onchange()

    from modules import sd_models
    sd_models.setup_model()
    startup_timer.record("setup SD model")

    initialize_rest(reload_script_modules=False)


def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    from modules.shared_cmd_options import cmd_opts

    from modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    from modules import extensions
    extensions.list_extensions()
    startup_timer.record("list extensions")

    from modules import initialize_util
    initialize_util.restore_config_state_file()
    startup_timer.record("restore config state file")

    from modules import shared, upscaler, scripts
    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        console.section("Extensions")
        with console.indent_output():
            scripts.load_scripts()
        return

    from modules import sd_models
    sd_models.list_models()
    startup_timer.record("list SD models")

    from modules import localization
    localization.list_localizations(cmd_opts.localizations_dir)
    startup_timer.record("list localizations")

    console.section("Extensions")
    with console.indent_output(), startup_timer.subcategory("load scripts"):
        scripts.load_scripts()

    from modules import sakuga_extras

    sakuga_extras.register()

    if reload_script_modules:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    from modules import modelloader
    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    from modules import sd_vae
    sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    from modules import sd_text_encoder
    sd_text_encoder.refresh_te_list()
    startup_timer.record("refresh text encoder")

    from modules import sd_unet
    sd_unet.list_unets()
    startup_timer.record("scripts list_unets")

    from modules import ui_extra_networks
    ui_extra_networks.initialize()
    ui_extra_networks.register_default_pages()

    from modules import extra_networks
    extra_networks.initialize()
    startup_timer.record("initialize extra networks")
