"""
LoRA Style Explorer - SD WebUI Script
Generates one image per LoRA from a list, switching LoRAs automatically.
Most of this code is from the X/Y/Z plot from AUTOMATIC1111/stable-diffusion-webui.
Works with auto complete like the sd-webui-tagcomplete extension.
Each entry starts at a <lora:...> tag and continues until the next tag, so commas inside trigger words are preserved, e.g. <lora:Example01:1.0> trigger_word, trigger_word2, <lora:OtherLora:0.75> trigger_word.
"""

import os
import re
from copy import copy

import gradio as gr
from PIL import Image, ImageColor, ImageDraw

import modules.images as images
import modules.scripts as scripts
from modules import errors, model_paths, shared
from modules.processing import Processed, process_images
from modules.shared import opts, state


# Extract LoRA name from snippet or tag (e.g. "<lora:StyleA:1.0> trigger" -> "StyleA")
RE_LORA_NAME = re.compile(r"<lora:\s*([^:>]+)", re.IGNORECASE)
# Full <lora:name> or <lora:name:weight> tag with name in group 1 (for selective removal)
RE_LORA_TAG = re.compile(r"<lora:\s*([^:>]+)(?::[^>]*)?>", re.IGNORECASE)


def extract_lora_name(snippet: str) -> str:
    m = RE_LORA_NAME.search(snippet.strip())
    return m.group(1).strip() if m else (snippet[:30] if snippet else "")


def prompt_with_lora_snippet(prompt: str, snippet: str, lora_names_in_list: set) -> str:
    names_lower = {n.strip().lower() for n in lora_names_in_list if n}

    def remove_only_list_lora(m):
        name = m.group(1).strip()
        if name.lower() in names_lower:
            return ""
        return m.group(0)

    cleaned = RE_LORA_TAG.sub(remove_only_list_lora, prompt or "")
    cleaned = " ".join(cleaned.split())  # normalize whitespace
    snippet = snippet.strip()
    return f"{cleaned} {snippet}".strip() if snippet else cleaned


def lora_entries_from_text(text: str):
    """Split at each LoRA tag; commas inside trigger words are preserved."""
    text = (text or "").strip()
    if not text:
        return []

    matches = list(RE_LORA_TAG.finditer(text))
    if not matches:
        return [text]

    entries = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entry = text[match.start():end].strip().strip(",").strip()
        if entry:
            entries.append(entry)
    return entries


def _wrap_text_to_width(draw, text, font, max_width, padding=4):
    """Split text into lines that fit within max_width. Returns list of lines."""
    if not text or max_width <= padding:
        return [text] if text else []
    lines = []
    words = text.split()
    line = []
    line_w = 0
    space_w = draw.textbbox((0, 0), " ", font=font)[2] - draw.textbbox((0, 0), " ", font=font)[0]
    for word in words:
        ww = draw.textbbox((0, 0), word, font=font)[2] - draw.textbbox((0, 0), word, font=font)[0]
        need_space = len(line)
        if line and line_w + (space_w if need_space else 0) + ww > max_width:
            lines.append(" ".join(line))
            line = [word]
            line_w = ww
        else:
            if need_space:
                line_w += space_w
            line.append(word)
            line_w += ww
    if line:
        lines.append(" ".join(line))
    # If a single line is still too long (e.g. one long word), break by character
    out = []
    for ln in lines:
        if draw.textbbox((0, 0), ln, font=font)[2] - draw.textbbox((0, 0), ln, font=font)[0] <= max_width:
            out.append(ln)
        else:
            for c in ln:
                if out and draw.textbbox((0, 0), out[-1] + c, font=font)[2] - draw.textbbox((0, 0), out[-1] + c, font=font)[0] <= max_width:
                    out[-1] += c
                else:
                    out.append(c)
    return out


def draw_grid_with_cell_labels(grid_img, cell_w, cell_h, labels_per_cell, margin=0):
    """Draw a grid with one label above each cell. labels_per_cell[col][row] = label text. Long text wraps within cell_w."""
    cols = len(labels_per_cell) if labels_per_cell else 0
    rows = len(labels_per_cell[0]) if cols else 0
    if rows == 0 or cols == 0:
        return grid_img
    fontsize = max(12, (cell_w + cell_h) // 25)
    try:
        font = images.get_font(fontsize)
    except Exception:
        font = None
    line_height = int(fontsize * 1.3) if font else 16
    max_label_lines = 3
    label_h = line_height * max_label_lines
    bg_color = ImageColor.getcolor(getattr(opts, "grid_background_color", "#ffffff"), "RGB")
    text_color = ImageColor.getcolor(getattr(opts, "grid_text_active_color", "#000000"), "RGB")
    total_w = cols * cell_w + (cols - 1) * margin if margin else cols * cell_w
    total_h = rows * (label_h + cell_h) + (rows - 1) * margin if margin else rows * (label_h + cell_h)
    result = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(result)
    max_text_w = max(1, cell_w - 8)
    for row in range(rows):
        for col in range(cols):
            text = labels_per_cell[col][row]
            y_row_start = row * (label_h + cell_h + margin)
            x_col_start = col * (cell_w + margin)
            if text and font:
                wrapped = _wrap_text_to_width(draw, text, font, max_text_w)
                for i, ln in enumerate(wrapped[:max_label_lines]):
                    bbox = draw.textbbox((0, 0), ln, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    tx = x_col_start + (cell_w - tw) // 2
                    ty = y_row_start + i * line_height + (line_height - th) // 2
                    draw.text((tx, ty), ln, font=font, fill=text_color)
            cell_img = grid_img.crop((col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h))
            result.paste(cell_img, (x_col_start, y_row_start + label_h))
    return result


def list_available_loras():
    try:
        candidates = set()
        for lora_dir in model_paths.lora_dirs(shared.cmd_opts.lora_dir):
            if not os.path.isdir(lora_dir):
                continue
            candidates.update(
                shared.walk_files(
                    lora_dir,
                    allowed_extensions=[".pt", ".ckpt", ".safetensors"],
                )
            )
        names = {os.path.splitext(os.path.basename(path))[0] for path in candidates if os.path.isfile(path)}
        return sorted(names, key=str.casefold)
    except Exception:
        return []


class Script(scripts.Script):
    def title(self):
        return "LoRA Style Explorer"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        with gr.Column():
            lora_list = gr.Textbox(
                label="LoRA list",
                placeholder="<lora:Name:1.0> trigger_word, trigger_word2, <lora:Other:0.8> ... (each entry starts at a LoRA tag)",
                lines=3,
                elem_id=self.elem_id("lora_list"),
                elem_classes=["prompt", "autocomplete"],
            )
            add_all_loras_btn = gr.Button(
                value="Add all LoRAs",
                variant="secondary",
                elem_id=self.elem_id("add_all_loras"),
            )

            def fill_all_loras():
                names = list_available_loras()
                if not names:
                    return ""
                return ", ".join(f"<lora:{n}:1.0>" for n in names)

            add_all_loras_btn.click(fn=fill_all_loras, outputs=[lora_list])

            create_grid = gr.Checkbox(
                label="Create grid (arrange all images in one grid)",
                value=True,
                elem_id=self.elem_id("create_grid"),
            )
            margin_size = gr.Slider(
                label="Grid margin (px)",
                minimum=0,
                maximum=100,
                step=2,
                value=0,
                elem_id=self.elem_id("margin_size"),
            )

        return [lora_list, create_grid, margin_size]

    def run(self, p, lora_list, create_grid, margin_size):
        lora_entries = lora_entries_from_text(lora_list or "")
        if not lora_entries:
            errors.report("LoRA Style Explorer: No LoRAs in list. Add entries starting with a <lora:...> tag.", exc_info=False)
            return Processed(p, [], p.seed, "")

        # Only remove LoRAs from the list, leave other LoRAs in the prompt as-is
        lora_names_in_list = {extract_lora_name(s) for s in lora_entries}

        if not opts.return_grid:
            p.batch_size = 1

        n_iter = p.n_iter
        total = len(lora_entries) * n_iter
        state.job_count = total

        all_images = []
        all_prompts = []
        all_seeds = []
        all_infotexts = []

        for i, snippet in enumerate(lora_entries):
            if state.interrupted or state.stopping_generation:
                break

            for n in range(n_iter):
                if state.interrupted or state.stopping_generation:
                    break

                job_index = i * n_iter + n
                state.job = f"LoRA {i + 1}/{len(lora_entries)} (image {job_index + 1}/{total})"

                pc = copy(p)
                pc.prompt = prompt_with_lora_snippet(p.prompt, snippet, lora_names_in_list)
                pc.n_iter = 1
                pc.do_not_save_grid = True

                try:
                    res = process_images(pc)
                except Exception:
                    errors.report(f"LoRA Style Explorer: Error for entry {i + 1}", exc_info=True)
                    continue

                if res and res.images:
                    for img in res.images:
                        all_images.append(img)
                        all_prompts.append(res.all_prompts[0] if res.all_prompts else pc.prompt)
                        all_seeds.append(res.all_seeds[0] if res.all_seeds else pc.seed)
                        all_infotexts.append(res.infotexts[0] if res.infotexts else "")

        if not all_images:
            return Processed(p, [], p.seed, "")

        # Build result
        if create_grid and len(all_images) > 1:
            rows = max(1, (len(all_images) + 4) // 5)  # up to 5 columns
            grid = images.image_grid(all_images, rows=rows)
            # Show LoRA name above each image (per-cell labels)
            try:
                grid_w, grid_h = all_images[0].size
                n_cols = (len(all_images) + rows - 1) // rows
                images_per_entry = max(1, len(all_images) // len(lora_entries))
                # labels_per_cell[col][row] = label for cell at (col, row); image index = row * n_cols + col
                labels_per_cell = []
                for col in range(n_cols):
                    col_labels = []
                    for row in range(rows):
                        idx = row * n_cols + col
                        if idx < len(all_images):
                            entry_index = min(idx // images_per_entry, len(lora_entries) - 1)
                            col_labels.append(extract_lora_name(lora_entries[entry_index]))
                        else:
                            col_labels.append("")
                    labels_per_cell.append(col_labels)
                grid = draw_grid_with_cell_labels(
                    grid, grid_w, grid_h, labels_per_cell, margin=margin_size or 0
                )
            except Exception:
                pass
            include_images = opts.return_grid
            result_images = [grid] + (all_images if include_images else [])
            result_prompts = all_prompts if include_images else all_prompts[:1]
            result_seeds = all_seeds if include_images else all_seeds[:1]
            result_infotexts = ([all_infotexts[0]] if all_infotexts else [""]) + (all_infotexts if include_images else [])
            index_of_first = 1
        else:
            result_images = all_images
            result_prompts = all_prompts
            result_seeds = all_seeds
            result_infotexts = all_infotexts
            index_of_first = 0

        processed = Processed(
            p,
            result_images,
            seed=result_seeds[0] if result_seeds else p.seed,
            info=result_infotexts[0] if result_infotexts else "",
            all_prompts=result_prompts,
            all_seeds=result_seeds,
            infotexts=result_infotexts,
            index_of_first_image=index_of_first,
        )

        if create_grid and len(all_images) > 1 and opts.grid_save:
            images.save_image(
                result_images[0],
                p.outpath_grids,
                "lora_explorer_grid",
                info=result_infotexts[0] if result_infotexts else "",
                extension=opts.grid_format,
                prompt=result_prompts[0] if result_prompts else "",
                seed=result_seeds[0] if result_seeds else p.seed,
                grid=True,
                p=p,
            )

        shared.total_tqdm.clear()
        return processed
