# XY Plot (Universal)

A drop-anywhere XY/XYZ-plot node for ComfyUI. Insert it in any workflow, configure values for upstream widgets, and it re-runs the relevant upstream sub-graph for every grid cell and composes a labeled plot image.

Original code from [TinyTerraNodes](https://github.com/tinyterra/ComfyUI_tinyterraNodes)

## Installation

Place the folder under `ComfyUI/custom_nodes`

## Inputs

| Name | Type | Description |
| ---- | ---- | ----------- |
| `image` | IMAGE | Required. The image captured for each grid cell. Wire from your VAE Decode (or any IMAGE source). |
| `grid_spacing` | INT | Pixels between cells. |
| `flip_xy` | BOOLEAN | Swap X and Y axes. |
| `show_preview` | BOOLEAN | Show the composed grid as a preview image inside the node. |
| `x_plot` | STRING (multiline) | Plot definition for the X axis. |
| `y_plot` | STRING (multiline) | Plot definition for the Y axis. |
| `z_plot` | STRING (multiline) | Plot definition for the Z axis (one grid image per Z step). |

## Outputs

| Name | Type | Description |
| ---- | ---- | ----------- |
| `plot_image` | IMAGE | The composed grid image(s). One per Z-step. |
| `images` | IMAGE | Individual cell images, or the grid if no axes are defined. |

## Plot syntax

```
<axis_step:label>
[NODE_ID:widget_name='value']
[NODE_ID:another_widget='another value']

<axis_step:label>
[NODE_ID:widget_name='another value']
```

- `axis_step` is an ordinal (`1`, `2`, `3`, …).
- `label` is a literal string, or one of `v_label` (values only), `tv_label` (`widget_name: value`), `idtv_label` (`[NODE_ID] widget_name: value`).
- Multiple `[…]` blocks under the same `<…>` header co-vary several widgets at once.
- Search/replace: `'%search;replacement%'` substitutes into the current widget value.
- Append: `widget_name.append` appends to the existing widget value.

## What it does in practicce

1. Walks the graph upstream, copying the sub-graph.
2. For each (Z, X, Y) combination, deep-copies the sub-graph, applies widget mutations, validates, and runs through a private executor.
3. Reads the IMAGE from the `image` input link per cell.
4. Composes the grid image