"""Native ImageStitch reference input for Anima Edit."""

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import images, scripts, shared
from modules.api import api
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img
from modules.sd_samplers_common import images_tensor_to_samples
from modules.ui_components import FormRow, InputAccordion
from modules_forge.anima_reference import reference as anima_reference


class ImageStitch(scripts.Script):
    sorting_priority = 529
    cached_parameters = None

    def title(self):
        return "ImageStitch Integrated"

    def show(self, is_img2img):
        if not getattr(shared.opts, "anima_do_reference", False):
            return False
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(
                "Use reference images with an Anima Edit LoRA. "
                "This does not stitch the images; each reference is encoded as an Anima reference latent."
            )
            references = gr.Gallery(
                value=None,
                type="pil",
                interactive=True,
                show_label=False,
                container=False,
                show_download_button=False,
                show_share_button=False,
                label="Reference Image(s)",
                min_width=512,
                height=384,
                columns=3,
                rows=1,
                allow_preview=False,
                object_fit="contain",
                elem_id=self.elem_id("ref_latent"),
            )

            select_index = gr.State(-1)

            def on_select(evt: gr.SelectData) -> int:
                return evt.index

            references.select(fn=on_select, outputs=[select_index], queue=False, show_progress=False)

            with FormRow():
                upload = gr.Image(
                    height=225,
                    width=225,
                    sources="upload",
                    type="pil",
                    label="Image to Upload",
                    show_download_button=False,
                    show_share_button=False,
                )
                with gr.Column():
                    btn_upload = gr.Button("Append Pasted Image")
                    btn_replace = gr.Button("Replace Selected Image")
                    btn_delete = gr.Button("Delete Selected Image", variant="stop")
                    btn_clear = gr.Button("Clear All References", variant="stop")

        def upload_image(gallery, image):
            if not image:
                return [gr.skip(), gr.skip()]
            gallery = list(gallery or [])
            gallery.append((image, None))
            return [gr.update(value=gallery), gr.update(value=None)]

        def replace_image(index, gallery, image):
            if not image or not gallery or index < 0 or index >= len(gallery):
                return [-1, gr.skip(), gr.skip()]
            gallery = list(gallery)
            gallery[index] = (image, None)
            return [-1, gr.update(value=gallery), gr.update(value=None)]

        def delete_image(index, gallery):
            if not gallery or index < 0 or index >= len(gallery):
                return [-1, gr.skip()]
            gallery = list(gallery)
            gallery.pop(index)
            return [-1, gr.update(value=gallery)]

        btn_upload.click(fn=upload_image, inputs=[references, upload], outputs=[references, upload], queue=False, show_progress=False)
        btn_replace.click(fn=replace_image, inputs=[select_index, references, upload], outputs=[select_index, references, upload], queue=False, show_progress=False)
        btn_delete.click(fn=delete_image, inputs=[select_index, references], outputs=[select_index, references], queue=False, show_progress=False)
        btn_clear.click(fn=lambda: [-1, gr.update(value=[])], outputs=[select_index, references], queue=False, show_progress=False)

        return [enable, references]

    @staticmethod
    def reset_references(p: StableDiffusionProcessing):
        p.cached_c = [None, None]
        p.cached_uc = [None, None]
        StableDiffusionProcessing.cached_c = p.cached_c
        StableDiffusionProcessing.cached_uc = p.cached_uc
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            p.cached_hr_c = [None, None]
            p.cached_hr_uc = [None, None]
            StableDiffusionProcessingTxt2Img.cached_hr_c = p.cached_hr_c
            StableDiffusionProcessingTxt2Img.cached_hr_uc = p.cached_hr_uc
        p.sd_model.clear_references()

    def process(self, p: StableDiffusionProcessing, enable: bool, references):
        if not getattr(p.sd_model, "is_anima", False):
            return

        if enable and references and getattr(shared.opts, "anima_do_reference", False):
            references = self.extract_images(references)
        else:
            references = None

        cache = None
        if references:
            cache = (
                getattr(shared.opts, "reference_max_side", 1024),
                getattr(p.sd_model, "sd_model_hash", ""),
                tuple(self.hash_image(ref) for ref in references),
            )

        if ImageStitch.cached_parameters == cache:
            return

        self.reset_references(p)
        ImageStitch.cached_parameters = cache

        if not references:
            return

        anima_reference.is_referencing = True
        try:
            for reference in references:
                image = images.flatten(reference, shared.opts.img2img_background_color)
                image = np.array(image, dtype=np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                image = torch.from_numpy(image).unsqueeze(0)
                images_tensor_to_samples(image.to(shared.device, dtype=torch.float32), 0, p.sd_model)
        finally:
            anima_reference.is_referencing = False

    @staticmethod
    def extract_images(gallery):
        if isinstance(gallery[0], str):
            return [api.decode_base64_to_image(img) for img in gallery]
        return [img for (img, _) in gallery]

    @staticmethod
    def hash_image(img: Image.Image) -> int:
        return hash((img.mode, img.size, img.convert("L").tobytes()))
