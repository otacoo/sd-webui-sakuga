"""Shared Anima Edit reference state.

Anima Edit LoRAs condition on reference latents concatenated with the noisy
latent.  The state is kept outside the model so the processing pipeline,
ImageStitch UI, and transformer forward pass can all reach it.
"""


class AnimaReference:
    def __init__(self):
        self.ref_latents = []
        self.ini_latent = None
        self.is_referencing = False

    def clear(self):
        self.ref_latents.clear()
        self.ini_latent = None
        self.is_referencing = False


reference = AnimaReference()
