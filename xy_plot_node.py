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
import random
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
# Plot text parsing - extended from tinyterraNodes' advanced xyPlot format.
# --------------------------------------------------------------------------- #
def _strip_comments(text):
    """Drop full-line `#` comments. Lines whose first non-whitespace char is
    `#` are removed; mid-line `#` is left alone (lora tags can legitimately
    contain it)."""
    return '\n'.join(
        line for line in text.splitlines()
        if not line.lstrip().startswith('#')
    )


def _render_label(label_kind, value_list):
    """Render an axis-step label from its raw values, given a label kind.
    Literal labels are returned unchanged."""
    if label_kind in ('v_label', 'tv_label', 'idtv_label'):
        parts = []
        for value, input_name, node_id in value_list:
            if label_kind == 'v_label':
                parts.append(str(value))
            elif label_kind == 'tv_label':
                parts.append(f'{input_name}: {value}')
            else:
                parts.append(f'[{node_id}] {input_name}: {value}')
        return ', '.join(parts)
    return label_kind


def _parse_plot_text(plot_data, axis_label="X"):
    """Parse a plot definition string into an OrderedDict keyed by axis-step.

    Each value is a dict::

        {
            "label": "<rendered text>",
            "_label_kind": "<literal>|v_label|tv_label|idtv_label",
            "_value_list": [(value, input_name, node_id), ...],
            "<node_id>": {"<widget_name>": "<value>", ...},
            ...
        }

    Supports:
      - `#` line comments (full-line)
      - `<:label>` auto-numbered axis headers
      - Per-widget value expressions (range/linspace/{...}/*) — expanded later
    """
    if plot_data is None or plot_data.strip() == '':
        return None

    plot_data = _strip_comments(plot_data)
    if plot_data.strip() == '':
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

        auto_num = 1
        for raw in merged:
            if not raw:
                continue
            head, _, body = raw.partition('>')
            if ':' not in head:
                continue
            num, label = head.split(':', 1)
            num = num.strip()
            if num == '':
                # `<:label>` - auto-number this header.
                num = str(auto_num)
            # Bump auto-num past whatever explicit number was used so future
            # `<:label>` headers don't collide with it.
            try:
                auto_num = max(auto_num, int(num) + 1)
            except ValueError:
                auto_num += 1

            axis_dict[num] = {"label": label, "_label_kind": label}

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

            axis_dict[num]["_value_list"] = values_label
            axis_dict[num]["label"] = _render_label(label, values_label)
    except ValueError:
        logging.warning(f"xyplot_universal: invalid {axis_label} plot - ignoring.")
        return None
    # Drop axis steps that have a header but no widget mutations under it.
    # The picker auto-appends a trailing <n:label> after every pick so the
    # next pick starts a fresh step; the last one is dangling until it gets
    # filled in. Treating it as a real step would plot a blank cell.
    axis_dict = OrderedDict(
        (k, v) for k, v in axis_dict.items() if v.get("_value_list")
    )
    return axis_dict


# --------------------------------------------------------------------------- #
# Range / list / combo expansion.
# A single axis step containing a `range(...)`, `linspace(...)`, `{a,b,c}`,
# `*`, `random_seed(n)` or `random(a,b,n)` expression on one or more widgets
# is expanded into multiple steps. If several widgets in the same step use
# expansions they are zipped together (lengths must match).
# --------------------------------------------------------------------------- #
_RE_RANGE = re.compile(
    r'^\s*range\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)'
    r'(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*\)\s*$'
)
_RE_LINSPACE = re.compile(
    r'^\s*linspace\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)'
    r'\s*,\s*(\d+)\s*\)\s*$'
)
_RE_RANDOM_SEED = re.compile(r'^\s*random_seed\s*\(\s*(\d+)\s*\)\s*$')
_RE_RANDOM = re.compile(
    r'^\s*random\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)'
    r'\s*,\s*(\d+)\s*\)\s*$'
)
_RE_LIST = re.compile(r'^\s*\{\s*(.+?)\s*\}\s*$')


def _is_int_widget(prompt, node_id, widget_name):
    """Best-effort: is this widget INT-typed? Used to pick int vs float for
    numeric range output. Falls back to float if unknown."""
    try:
        ct = prompt[node_id]['class_type']
        td = COMFY_CLASS_MAPPINGS[ct].INPUT_TYPES()
        for itype in ('required', 'optional'):
            d = td.get(itype) or {}
            if widget_name in d and d[widget_name]:
                return d[widget_name][0] == 'INT'
    except Exception:
        pass
    return False


def _combo_values(prompt, node_id, widget_name):
    """Return the list of legal combo values for a widget, or None."""
    try:
        ct = prompt[node_id]['class_type']
        td = COMFY_CLASS_MAPPINGS[ct].INPUT_TYPES()
        for itype in ('required', 'optional'):
            d = td.get(itype) or {}
            if widget_name in d and d[widget_name]:
                kind = d[widget_name][0]
                if isinstance(kind, list):
                    return [str(v) for v in kind]
    except Exception:
        pass
    return None


def _frange(start, stop, step):
    """Python-style numeric range supporting floats and negative steps."""
    if step == 0:
        raise ValueError("range step cannot be zero")
    out = []
    n = start
    # Use a small epsilon-free loop driven by an index to avoid drift.
    i = 0
    while True:
        v = start + i * step
        if step > 0 and v >= stop:
            break
        if step < 0 and v <= stop:
            break
        out.append(v)
        i += 1
        if i > 10000:
            raise ValueError("range expands to too many values (>10000)")
    return out


def _format_number(v, as_int):
    if as_int:
        return str(int(round(v)))
    # Trim trailing zeros for readability while keeping floats precise.
    s = f"{v:.6f}".rstrip('0').rstrip('.')
    return s or '0'


def _try_expand_value(value, prompt, node_id, widget_name):
    """If `value` is a range/list/combo-star expression, return a list of
    expanded string values. Otherwise return None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None

    # Combo wildcard
    if v == '*':
        return _combo_values(prompt, node_id, widget_name)

    m = _RE_RANGE.match(v)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        step = float(m.group(3)) if m.group(3) is not None else None
        if step is None:
            # Sensible default: int widgets step by 1, floats by (b-a)/10.
            step = 1.0 if _is_int_widget(prompt, node_id, widget_name) else (b - a) / 10.0
        as_int = (
            _is_int_widget(prompt, node_id, widget_name)
            and step == int(step) and a == int(a) and b == int(b)
        )
        nums = _frange(a, b, step)
        return [_format_number(x, as_int) for x in nums]

    m = _RE_LINSPACE.match(v)
    if m:
        a = float(m.group(1)); b = float(m.group(2)); n = int(m.group(3))
        if n <= 0:
            return []
        if n == 1:
            return [_format_number(a, _is_int_widget(prompt, node_id, widget_name))]
        as_int = _is_int_widget(prompt, node_id, widget_name)
        step = (b - a) / (n - 1)
        return [_format_number(a + i * step, as_int) for i in range(n)]

    m = _RE_RANDOM_SEED.match(v)
    if m:
        n = int(m.group(1))
        return [str(random.randint(0, 2**31 - 1)) for _ in range(n)]

    m = _RE_RANDOM.match(v)
    if m:
        a = float(m.group(1)); b = float(m.group(2)); n = int(m.group(3))
        as_int = _is_int_widget(prompt, node_id, widget_name)
        if as_int:
            lo, hi = int(min(a, b)), int(max(a, b))
            return [str(random.randint(lo, hi)) for _ in range(n)]
        return [_format_number(random.uniform(a, b), False) for _ in range(n)]

    m = _RE_LIST.match(v)
    if m:
        inner = m.group(1)
        # Split on commas; preserve internal whitespace inside items but
        # trim each item's outer whitespace.
        items = [p.strip() for p in inner.split(',')]
        items = [p for p in items if p != '']
        return items if items else None

    return None


def _expand_axis(axis_dict, prompt, axis_label="X"):
    """Expand any range/list/combo expressions in an axis dict. Multi-widget
    expansions in the same step are zipped (lengths must match). Steps with
    no expansion are passed through unchanged."""
    if axis_dict is None:
        return None
    out = OrderedDict()
    next_step = 1
    for step_key, entry in axis_dict.items():
        expansions = []   # list of (node_id, widget_name, [values])
        static = OrderedDict()  # node_id -> OrderedDict(widget_name -> value)
        for node_id, widget_inputs in entry.items():
            if node_id in ("label", "_label_kind", "_value_list"):
                continue
            for w, val in widget_inputs.items():
                exp = _try_expand_value(val, prompt, node_id, w)
                if exp is not None:
                    expansions.append((node_id, w, exp))
                else:
                    static.setdefault(node_id, OrderedDict())[w] = val

        if not expansions:
            new_entry = copy.deepcopy(entry)
            out[str(next_step)] = new_entry
            next_step += 1
            continue

        lengths = {len(exp) for _, _, exp in expansions}
        if len(lengths) > 1:
            raise ValueError(
                f"xyplot_universal: {axis_label} axis step {step_key} has "
                f"expansions of mismatched lengths: {sorted(lengths)}. "
                "When multiple widgets in one step use range/list, they must "
                "produce the same number of values (they are zipped)."
            )
        n = lengths.pop()
        if n == 0:
            continue  # empty expansion - skip the step entirely

        label_kind = entry.get("_label_kind", entry.get("label", ""))

        for i in range(n):
            new_entry = {"_label_kind": label_kind}
            value_list = []
            for nid, ws in static.items():
                new_entry[nid] = dict(ws)
                for wn, wv in ws.items():
                    value_list.append((wv, wn, nid))
            for nid, w, exp in expansions:
                new_entry.setdefault(nid, {})[w] = exp[i]
                value_list.append((exp[i], w, nid))
            new_entry["_value_list"] = value_list
            new_entry["label"] = _render_label(label_kind, value_list)
            out[str(next_step)] = new_entry
            next_step += 1
    return out


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
            if node_id in ('label', '_label_kind', '_value_list'):
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

        # Expand range/list/combo expressions into individual steps. Done
        # against the live prompt so widget types (INT vs FLOAT, combo legal
        # values) can drive expansion decisions.
        x_points = _expand_axis(x_points, prompt, "X")
        y_points = _expand_axis(y_points, prompt, "Y")
        z_points = _expand_axis(z_points, prompt, "Z")

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
