# XY Plot (Universal)

A drop-anywhere XY/XYZ-plot node for ComfyUI. Insert it in any workflow, configure values for upstream widgets, and it re-runs the relevant upstream sub-graph for every grid cell and composes a labeled plot image.

Originally adapted from [TinyTerraNodes](https://github.com/tinyterra/ComfyUI_tinyterraNodes); extended with range expressions, filter-as-you-type picker, an in-node help button, and V3-node support.

## Installation

Place the folder under `ComfyUI/custom_nodes`.

## Plot syntax

### Basic shape

```
<1:v_label>
[5:steps='20']
[5:cfg='7.5']

<2:v_label>
[5:steps='40']
[5:cfg='5.0']
```

- `<n:label>` — axis-step header. `n` is the step ordinal.
- `<:label>` — leave the number blank to auto-number.
- `[NODE_ID:widget='value']` — set `widget` on node `NODE_ID`.
- Multiple `[...]` lines under one header co-vary several widgets in the same cell.
- `# comment` — full-line comments are ignored.

### Label kinds

| Kind | Result |
| ---- | ------ |
| `v_label` | values only — `20, 7.5` |
| `tv_label` | title + value — `steps: 20, cfg: 7.5` |
| `idtv_label` | id + title + value — `[5] steps: 20, [5] cfg: 7.5` |
| anything else | used as a literal label |

### Range expressions

Each of these expands one axis-step into many cells:

```
<1:v_label>
[5:steps=range(10, 40, 10)]      # → 10, 20, 30
[5:cfg=linspace(1.0, 10.0, 5)]   # → 1.0, 3.25, 5.5, 7.75, 10.0
[5:sampler_name='*']             # → every legal sampler (combo widgets only)
[5:scheduler='{karras, simple, normal}']
[5:seed=random_seed(4)]          # → 4 fresh random seeds
[5:cfg=random(1.0, 10.0, 5)]     # → 5 random floats in [1, 10]
```

| Form | Meaning |
| ---- | ------- |
| `range(a, b, step)` | Python-style, end-exclusive. Step optional (defaults: 1 for ints, `(b-a)/10` for floats). |
| `linspace(a, b, n)` | `n` evenly-spaced values, endpoints included. |
| `{a, b, c}` | Explicit list. |
| `*` | All legal values of a combo widget. |
| `random_seed(n)` | `n` fresh random seed values. |
| `random(a, b, n)` | `n` random values in `[a, b]` (int or float, picked from widget type). |

If several widgets in one step use ranges, they are **zipped** (lengths must match). No Cartesian explosion — use multiple axis steps if you want that.

### Text tricks

- `widget.append` — appends to the existing value: `[7:text.append=', cinematic']`
- `'%search;replace%'` — substitute into the existing value: `[7:text='%dog;cat%']`

## What it does in practice

1. Walks the graph upstream, copying the sub-graph.
2. Parses each axis textbox, then expands range/list/combo expressions into individual steps.
3. For each (Z, X, Y) cell, deep-copies the sub-graph, applies widget mutations, validates, and runs through a private executor.
4. Reads the IMAGE from the `image` input link per cell.
5. Composes the labeled grid image.
