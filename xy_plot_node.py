"""
Universal XY Plot node for ComfyUI.

This node can be inserted anywhere in a workflow (no dependency on any
particular sampler / pipe). On execution, it walks every node upstream of
itself, makes deep copies of that sub-graph, mutates widget values according
to the user-supplied X/Y/Z plot definitions, re-executes the sub-graph for
every grid cell, captures the IMAGE input on every iteration, and finally
composes a labeled grid image.

The node accepts:

  required:
    image            - IMAGE  - the image that should be captured per cell
                                (typically wired from the VAE Decode output).
    grid_spacing     - INT
    flip_xy          - BOOLEAN
    show_preview     - BOOLEAN (default False) - when True, the composed grid
                                is also shown as a preview image inside the
                                node itself.
    x_plot           - STRING (multiline)
    y_plot           - STRING (multiline)
    z_plot           - STRING (multiline)

The plot text format mirrors the tinyterraNodes "advanced xyPlot" syntax:

  <axis_number:label_kind>
  [node_id:widget_name='value']
  [node_id:widget_name='value']

  <axis_number:label_kind>
  [node_id:widget_name='value']
  ...

`label_kind` may be a literal string used as the label, or one of
`v_label` (values), `tv_label` (title and values), `idtv_label`
(node id, title and values).

Values support a search/replace mode: '%search;replacement%'.
You can also append to an existing widget value with `widget_name.append`.
"""

import copy
import os
import re
import uuid
import asyncio
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import nodes
import execution
import folder_paths
from nodes import NODE_CLASS_MAPPINGS as COMFY_CLASS_MAPPINGS

from .executor import SubGraphExecutor, register_intercept


CLASS_TYPE = "XYPlotUniversal"


# --------------------------------------------------------------------------- #
# Any-type helper - lets a socket accept / produce any link type.
# --------------------------------------------------------------------------- #
class AnyType(str):
    """A string that compares equal to anything - used so ComfyUI accepts a
    connection of any type on this socket."""
    def __eq__(self, _other):
        return True
    def __ne__(self, _other):
        return False
    def __hash__(self):
        return hash(str(self))


ANY_TYPE = AnyType("*")


# --------------------------------------------------------------------------- #
# Plot text parsing - same format as tinyterraNodes' advanced xyPlot.
# --------------------------------------------------------------------------- #
def _parse_plot_text(plot_data, axis_label="X"):
    """Parse a plot definition string into an OrderedDict keyed by axis-step.

    Each value is a dict::

        {
            "label": "<text>",
            "<node_id>": {"<widget_name>": "<value>", ...},
            ...
        }
    """
    if plot_data is None or plot_data.strip() == '':
        return None

    try:
        axis_dict = OrderedDict()
        # Allow unescaped '<' inside lora references such as <lora:foo:1.0>.
        lines = plot_data.split('<')
        merged = []
        for line in lines:
            if line.startswith('lora') and merged:
                merged[-1] += '<' + line
            else:
                merged.append(line)

        for raw in merged:
            if not raw:
                continue
            head, _, body = raw.partition('>')
            if ':' not in head:
                continue
            num, label = head.split(':', 1)
            num = num.strip()
            axis_dict[num] = {"label": label}

            values_label = []
            for chunk in body.split('['):
                chunk = chunk.strip()
                if not chunk or ':' not in chunk:
                    continue
                # Strip trailing ']' and stray whitespace.
                chunk = chunk.rstrip(']').strip()
                node_id, _, rest = chunk.partition(':')
                node_id = node_id.strip()
                if '=' not in rest:
                    continue
                input_name, _, value_part = rest.partition('=')
                input_name = input_name.strip()
                # Value is the text between the first pair of quotes.
                if "'" in value_part:
                    value = value_part.split("'", 2)[1]
                else:
                    value = value_part.strip()
                axis_dict[num].setdefault(node_id, {})[input_name] = value
                values_label.append((value, input_name, node_id))

            if label in ('v_label', 'tv_label', 'idtv_label'):
                rendered = []
                for value, input_name, node_id in values_label:
                    if label == 'v_label':
                        rendered.append(value)
                    elif label == 'tv_label':
                        rendered.append(f'{input_name}: {value}')
                    else:
                        rendered.append(f'[{node_id}] {input_name}: {value}')
                axis_dict[num]['label'] = ', '.join(rendered)
    except ValueError:
        logging.warning(f"xyplot_universal: invalid {axis_label} plot - ignoring.")
        return None
    return axis_dict


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


_FONT_PATH_CANDIDATES = [
    Path(__file__).parent / "arial.ttf",
    Path(__file__).parent.parent / "arial.ttf",  # parent project bundles arial.ttf
]


def _load_font(size):
    for p in _FONT_PATH_CANDIDATES:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Fake class used to short-circuit our own node when the sub-executor walks
# the upstream graph - it returns immediately so we don't recurse forever.
# --------------------------------------------------------------------------- #
class _PlotPassthroughFake:
    """Replacement class injected into the sub-executor. It mirrors the real
    node's input/output signature but returns dummy outputs without running
    any plot logic. The actual IMAGE output is irrelevant - we read the image
    from the upstream node directly via the prompt link."""

    @classmethod
    def INPUT_TYPES(cls):
        # Mirror only what we need; sub-executor doesn't validate strictly.
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "noop"

    def noop(self, image=None, **_kwargs):
        # Return something compatible; outputs of our own node are never read
        # by the sub-graph (we are the sink).
        empty = torch.zeros((1, 1, 1, 3))
        if image is None:
            image = empty
        return (image, empty)


register_intercept(CLASS_TYPE, _PlotPassthroughFake)


# --------------------------------------------------------------------------- #
# The actual node
# --------------------------------------------------------------------------- #
class XYPlotUniversalNode:
    """Standalone universal XY-plot node - drop it anywhere in a workflow."""

    PLOT_PLACEHOLDER = (
        "_PLOT\n"
        "Example:\n\n"
        "<1:label1>\n"
        "[NODE_ID:widget_name='value1']\n\n"
        "<2:label2>\n"
        "[NODE_ID:widget_name='value2']\n"
        "[NODE_ID:widget2_name='value']\n"
        "[NODE_ID2:widget_name='value']\n\n"
        "etc..."
    )

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_spacing": ("INT", {"min": 0, "max": 500, "step": 5, "default": 0}),
                "flip_xy": ("BOOLEAN", {"default": False}),
                "show_preview": ("BOOLEAN", {"default": False}),
                "x_plot": ("STRING", {
                    "default": '', "multiline": True,
                    "placeholder": 'X' + cls.PLOT_PLACEHOLDER,
                    "pysssss.autocomplete": False,
                }),
                "y_plot": ("STRING", {
                    "default": '', "multiline": True,
                    "placeholder": 'Y' + cls.PLOT_PLACEHOLDER,
                    "pysssss.autocomplete": False,
                }),
                "z_plot": ("STRING", {
                    "default": '', "multiline": True,
                    "placeholder": 'Z' + cls.PLOT_PLACEHOLDER,
                    "pysssss.autocomplete": False,
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("plot_image", "images")
    FUNCTION = "plot"
    OUTPUT_NODE = True
    CATEGORY = "xyplot_universal"

    # ------------------------------------------------------------------ #
    # Sub-graph extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect_upstream_nodes(node_id, prompt):
        """Return ordered list of upstream node ids leading into `node_id`.

        Order is roughly leaves-first so the resulting prompt can be executed
        in order (the sub-executor itself handles dependencies but this keeps
        things tidy)."""
        keep = OrderedDict([(node_id, None)])
        stack = [node_id]
        while stack:
            cur = stack.pop()
            for v in prompt[cur]["inputs"].values():
                if isinstance(v, list) and len(v) == 2:
                    pid = v[0]
                    if pid not in keep and pid in prompt:
                        keep[pid] = None
                        stack.append(pid)
        return list(reversed(list(keep.keys())))

    def _build_base_prompt(self, prompt, my_unique_id):
        keep = self._collect_upstream_nodes(my_unique_id, prompt)
        return {nid: prompt[nid] for nid in keep}

    # ------------------------------------------------------------------ #
    # Value parsing & substitution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_value(input_name, value, node_inputs, input_types, regex):
        # `widget.append` -> append to existing string value.
        if input_name.endswith('.append'):
            input_name = input_name[: -len('.append')]
            value = str(node_inputs.get(input_name, '')) + ' ' + str(value)

        # Search/replace mode: %search;replace%
        if isinstance(value, str):
            matches = regex.findall(value)
            if matches:
                value = node_inputs.get(input_name, '')
                for search, replace in matches:
                    pattern = re.compile(re.escape(search), re.IGNORECASE)
                    value = pattern.sub(replace, value)

        # Type-coerce based on the destination widget definition.
        for itype in ('required', 'optional'):
            for iname, ivalues in (input_types.get(itype) or {}).items():
                if iname != input_name:
                    continue
                if not ivalues:
                    continue
                kind = ivalues[0]
                # Empty-string values for numeric/boolean widgets mean
                # "leave existing value alone" - this avoids hard-crashing
                # the whole plot if the user left a placeholder behind.
                is_empty = isinstance(value, str) and value.strip() == ''
                if kind == 'INT':
                    if is_empty:
                        value = node_inputs.get(input_name, 0)
                    else:
                        try:
                            value = int(float(value))
                        except (TypeError, ValueError):
                            logging.warning(
                                f"xyplot_universal: invalid INT value '{value}' "
                                f"for '{input_name}' - keeping existing value.")
                            value = node_inputs.get(input_name, 0)
                elif kind == 'FLOAT':
                    if is_empty:
                        value = node_inputs.get(input_name, 0.0)
                    else:
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            logging.warning(
                                f"xyplot_universal: invalid FLOAT value '{value}' "
                                f"for '{input_name}' - keeping existing value.")
                            value = node_inputs.get(input_name, 0.0)
                elif kind in ('BOOL', 'BOOLEAN'):
                    if isinstance(value, str):
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif is_empty:
                            value = node_inputs.get(input_name, False)
                    value = bool(value)
                elif isinstance(kind, list):
                    if is_empty:
                        # Skip silently - leave the existing combo selection.
                        value = node_inputs.get(input_name, kind[0] if kind else '')
                    elif value not in kind:
                        raise KeyError(
                            f'"{value}" not a valid value for input "{iname}" in xyplot')
        return input_name, value

    def _apply_axis_mutations(self, prompt, axis_dict_entry, regex):
        for node_id, inputs in axis_dict_entry.items():
            if node_id == 'label':
                continue
            if node_id not in prompt:
                raise KeyError(
                    f'Node id "{node_id}" referenced by xyplot was not found '
                    f'upstream of the plot node.')
            node_inputs = prompt[node_id]['inputs']
            class_type = prompt[node_id]['class_type']
            class_def = COMFY_CLASS_MAPPINGS[class_type]
            input_types = class_def.INPUT_TYPES()
            for input_name, value in inputs.items():
                resolved_name, resolved_value = self._coerce_value(
                    input_name, value, node_inputs, input_types, regex)
                node_inputs[resolved_name] = resolved_value
        return prompt

    # ------------------------------------------------------------------ #
    # Sub-graph execution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_prompt(prompt_id, prompt):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import threading
            box = {}
            def runner():
                box['result'] = asyncio.run(execution.validate_prompt(prompt_id, prompt, None))
            t = threading.Thread(target=runner)
            t.start()
            t.join()
            return box['result']
        return loop.run_until_complete(execution.validate_prompt(prompt_id, prompt, None))

    def _execute_one(self, executor, prompt, extra_data, my_unique_id, label_text):
        prompt_id = uuid.uuid4()
        valid = self._validate_prompt(prompt_id, prompt)
        if not valid[0]:
            raise Exception(valid[1])

        logging.info(f"xyplot_universal: executing cell -> {label_text}")
        # Validate returns the list of output-node ids to execute. We only
        # need to drive execution from OUR plot node so the upstream graph
        # re-runs deterministically (any extra OUTPUT_NODEs upstream will be
        # picked up via dependency walking when they happen to lie on the
        # path leading into our IMAGE input).
        targets = [my_unique_id] if my_unique_id in valid[2] else list(valid[2])
        executor.execute(prompt, prompt_id, extra_data, targets)

        # Read the IMAGE input link of our own node from the executor's outputs.
        node_inputs = prompt[my_unique_id]['inputs']
        if 'image' not in node_inputs or not isinstance(node_inputs['image'], list):
            raise Exception(
                "xyplot_universal: 'image' input must be connected to an "
                "IMAGE-producing node (e.g. VAE Decode).")
        link_node_id, link_index = node_inputs['image']
        if link_node_id not in executor.outputs:
            raise Exception(
                f"xyplot_universal: upstream node {link_node_id} did not produce "
                f"an output during the plot iteration.")
        image_tensor = executor.outputs[link_node_id][link_index][0]
        return tensor2pil(image_tensor)

    # ------------------------------------------------------------------ #
    # Grid composition
    # ------------------------------------------------------------------ #
    @staticmethod
    def _adjust_font_size(text, initial, label_width):
        font = _load_font(initial)
        try:
            l, _t, r, _b = font.getbbox(text)
            text_width = r - l
        except Exception:
            text_width = font.getlength(text)
        scale = 0.9
        if text_width > label_width * scale and text_width > 0:
            return max(8, int(initial * (label_width / text_width) * scale))
        return initial

    @classmethod
    def _make_label(cls, width_or_height, text, initial_font_size, fill_color,
                    is_x_label=True, max_font=70, min_font=20):
        label_width = width_or_height
        size = cls._adjust_font_size(text, initial_font_size, label_width)
        size = max(min_font, min(max_font, size))
        font = _load_font(size)

        # Word wrap
        d_tmp = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        def measure(s):
            try:
                return d_tmp.textlength(s, font=font)
            except Exception:
                l, _t, r, _b = font.getbbox(s)
                return r - l

        words = text.split() if text else ['']
        if not words:
            words = ['']
        lines = []
        cur = words[0]
        for w in words[1:]:
            if measure(f"{cur} {w}") <= label_width:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)

        line_h = int(size * 1.2)
        total_h = max(line_h, len(lines) * line_h)

        img = Image.new('RGBA', (label_width, total_h), color=(0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        y = 0
        for line in lines:
            tw = measure(line)
            x = max(0, (label_width - int(tw)) // 2)
            d.text((x, y), line, fill=fill_color, font=font)
            y += line_h
        return img

    def _compose_grid(self, image_list, x_labels, y_labels, z_label,
                      num_cols, num_rows, grid_spacing, has_x, has_y, has_z):
        if not image_list:
            return None

        max_w = max(im.width for im in image_list)
        max_h = max(im.height for im in image_list)

        border = int((max_w // 8) * 1.5) if (has_x or has_y or has_z) else 0
        bg_w = num_cols * (max_w + grid_spacing) - grid_spacing + (border if has_y else 0)
        bg_h = (num_rows * (max_h + grid_spacing) - grid_spacing
                + (border if has_x else 0)
                + (border if has_z else 0))

        x_offset_initial = border if has_y else 0
        y_offset = border if has_x else 0

        # Always use white background with black labels.
        bg_color = (255, 255, 255, 255)
        fill_color = 'black'

        bg = Image.new('RGBA', (bg_w, bg_h), color=bg_color)

        for r in range(num_rows):
            x_offset = x_offset_initial
            for c in range(num_cols):
                idx = c * num_rows + r
                if idx >= len(image_list):
                    continue
                img = image_list[idx]
                bg.paste(img, (x_offset, y_offset))

                if r == 0 and has_x and c < len(x_labels):
                    lbl = self._make_label(
                        img.width, x_labels[c], int(48 * img.width / 512),
                        fill_color, is_x_label=True)
                    ly = max(0, (y_offset - lbl.height) // 2)
                    bg.alpha_composite(lbl, (x_offset, ly))

                if c == 0 and has_y and r < len(y_labels):
                    lbl = self._make_label(
                        img.height, y_labels[r], int(48 * img.height / 512),
                        fill_color, is_x_label=False)
                    lbl = lbl.rotate(90, expand=True)
                    lx = max(0, (x_offset - lbl.width) // 2)
                    ly = y_offset + max(0, (img.height - lbl.height) // 2)
                    bg.alpha_composite(lbl, (lx, ly))

                if z_label is not None and has_z:
                    lbl = self._make_label(
                        bg_w, z_label, int(48 * img.height / 512), fill_color)
                    ly = bg_h - lbl.height - lbl.height // 2
                    bg.alpha_composite(lbl, (0, ly))

                x_offset += img.width + grid_spacing
            y_offset += max_h + grid_spacing

        return pil2tensor(bg)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def plot(self, image, grid_spacing, flip_xy, show_preview,
             x_plot, y_plot, z_plot,
             prompt=None, extra_pnginfo=None, my_unique_id=None):

        x_points = _parse_plot_text(x_plot, "X")
        y_points = _parse_plot_text(y_plot, "Y")
        z_points = _parse_plot_text(z_plot, "Z")
        if x_points == {} or x_points is None:
            x_points = None
        if y_points == {} or y_points is None:
            y_points = None
        if z_points == {} or z_points is None:
            z_points = None

        if flip_xy:
            x_points, y_points = y_points, x_points

        # Nothing configured -> just pass the input through.
        if x_points is None and y_points is None:
            return {
                "ui": {"images": []},
                "result": (image, image),
            }

        my_unique_id = str(my_unique_id)
        if prompt is None or my_unique_id not in prompt:
            raise Exception(
                "xyplot_universal: cannot access workflow prompt for "
                "sub-graph execution.")

        base_prompt = self._build_base_prompt(prompt, my_unique_id)
        regex = re.compile(r'%(.*?);(.*?)%')

        num_cols = len(x_points) if x_points else 1
        num_rows = len(y_points) if y_points else 1

        executor = SubGraphExecutor()
        image_list = []
        x_labels = []
        y_labels = []
        per_z_grids = []

        z_iter = z_points.items() if z_points else [('1', {'label': None})]
        has_x = x_points is not None
        has_y = y_points is not None
        has_z = z_points is not None

        try:
            for _, z_entry in z_iter:
                z_label = z_entry.get('label')
                z_prompt = copy.deepcopy(base_prompt)
                z_prompt = self._apply_axis_mutations(z_prompt, z_entry, regex)

                cell_images = []

                if has_x:
                    x_labels = []
                    for _, x_entry in x_points.items():
                        x_label = x_entry.get('label')
                        x_labels.append(x_label)
                        x_prompt = copy.deepcopy(z_prompt)
                        x_prompt = self._apply_axis_mutations(x_prompt, x_entry, regex)

                        if has_y:
                            y_labels = []
                            for _, y_entry in y_points.items():
                                y_label = y_entry.get('label')
                                y_labels.append(y_label)
                                y_prompt = copy.deepcopy(x_prompt)
                                y_prompt = self._apply_axis_mutations(y_prompt, y_entry, regex)
                                pil = self._execute_one(
                                    executor, y_prompt,
                                    {'extra_pnginfo': extra_pnginfo},
                                    my_unique_id,
                                    f"X={x_label} Y={y_label} Z={z_label}")
                                cell_images.append(pil)
                        else:
                            pil = self._execute_one(
                                executor, x_prompt,
                                {'extra_pnginfo': extra_pnginfo},
                                my_unique_id,
                                f"X={x_label} Z={z_label}")
                            cell_images.append(pil)
                elif has_y:
                    y_labels = []
                    for _, y_entry in y_points.items():
                        y_label = y_entry.get('label')
                        y_labels.append(y_label)
                        y_prompt = copy.deepcopy(z_prompt)
                        y_prompt = self._apply_axis_mutations(y_prompt, y_entry, regex)
                        pil = self._execute_one(
                            executor, y_prompt,
                            {'extra_pnginfo': extra_pnginfo},
                            my_unique_id,
                            f"Y={y_label} Z={z_label}")
                        cell_images.append(pil)

                image_list = cell_images
                grid_tensor = self._compose_grid(
                    image_list, x_labels, y_labels, z_label,
                    num_cols, num_rows, grid_spacing,
                    has_x, has_y, has_z)
                if grid_tensor is not None:
                    per_z_grids.append(grid_tensor)
        finally:
            executor.reset()
            del executor

        if not per_z_grids:
            return {
                "ui": {"images": []},
                "result": (image, image),
            }

        plot_out = torch.cat(per_z_grids, dim=0)
        individuals_out = torch.cat([pil2tensor(p) for p in image_list], dim=0) \
            if image_list else plot_out

        # Only save the preview image when the toggle is on.
        ui_images = self._save_preview(plot_out) if show_preview else []

        return {"ui": {"images": ui_images}, "result": (plot_out, individuals_out)}

    # ------------------------------------------------------------------ #
    # Saving preview
    # ------------------------------------------------------------------ #
    @staticmethod
    def _save_preview(images_tensor):
        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for i in range(images_tensor.shape[0]):
            arr = images_tensor[i].cpu().numpy()
            pil = Image.fromarray(np.clip(255. * arr, 0, 255).astype(np.uint8))
            unique = uuid.uuid4().hex[:8]
            filename = f"xyplot_universal_{unique}_{i}.png"
            pil.save(os.path.join(out_dir, filename))
            results.append({"filename": filename, "subfolder": "", "type": "temp"})
        return results
