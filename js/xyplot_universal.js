/*
 * Frontend helper for the XY Plot (Universal) node.
 */

import { app } from "../../scripts/app.js";

const NODE_CLASS = "XYPlotUniversal";

const WIDGETS_TO_IGNORE = new Set([
    "control_after_generate",
    "empty_latent_aspect",
    "empty_latent_width",
    "empty_latent_height",
    "batch_size",
]);

/* --------------------------------------------------------------- */
/* CSS                                                             */
/* --------------------------------------------------------------- */
const STYLE_ID = "xyplot-universal-style";
function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.xyplot-universal-dropdown,
.xyplot-universal-dropdown ul {
    position: relative;
    box-sizing: border-box;
    background-color: #171717;
    box-shadow: 0 4px 4px rgba(255, 255, 255, .25);
    padding: 0;
    margin: 0;
    list-style: none;
    z-index: 10000;
    overflow: visible;
    max-height: fit-content;
    max-width: fit-content;
}
.xyplot-universal-dropdown {
    position: absolute;
    border-radius: 0;
    color: #ddd;
}
.xyplot-universal-dropdown li,
.xyplot-universal-dropdown ul li {
    padding: 4px 14px 4px 10px;
    cursor: pointer;
    font-family: system-ui;
    font-size: 0.7rem;
    position: relative;
    white-space: nowrap;
}
.xyplot-universal-dropdown li.item,
.xyplot-universal-dropdown ul li.item {
    font-weight: normal;
    min-width: max-content;
}
.xyplot-universal-dropdown li.folder,
.xyplot-universal-dropdown ul li.folder {
    cursor: default;
    border-right: 3px solid #005757;
    padding-right: 18px;
}
.xyplot-universal-dropdown li.folder::after,
.xyplot-universal-dropdown ul li.folder::after {
    content: ">";
    position: absolute;
    right: 4px;
    font-weight: normal;
}
.xyplot-universal-dropdown ul {
    position: absolute;
    top: 0;
    left: 100%;
    border: none;
    display: none;
}
.xyplot-universal-dropdown li.selected > ul,
.xyplot-universal-dropdown ul li.selected > ul {
    display: block;
    border: none;
}
.xyplot-universal-dropdown li.selected,
.xyplot-universal-dropdown ul li.selected {
    background-color: #222222;
    border: none;
}

/* ---------- 50/50 layout for plot node ---------- */
.xyplot-grid {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 2px !important;
    align-content: start !important;
}
.xyplot-grid > [data-xyplot-role="left"] {
    grid-column: 1 !important;
}
.xyplot-grid > [data-xyplot-role="right"] {
    grid-column: 2 !important;
    grid-row: 2 / -1 !important;
    align-self: start !important;
    justify-self: center !important;
    max-width: 100% !important;
    overflow: hidden !important;
}
.xyplot-grid > [data-xyplot-role="right"] img,
.xyplot-grid > [data-xyplot-role="right"] canvas {
    max-width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}
.xyplot-grid > [data-xyplot-role="full"] {
    grid-column: 1 / -1 !important;
}
`;
    document.head.appendChild(style);
}

/* --------------------------------------------------------------- */
/* Dropdown                                                        */
/* --------------------------------------------------------------- */
let _activeDropdown = null;
let _activeOnDocClick = null;
let _activeAnchor = null;
function _removeDropdown() {
    if (_activeDropdown && _activeDropdown.parentNode) {
        _activeDropdown.parentNode.removeChild(_activeDropdown);
    }
    _activeDropdown = null;
    _activeAnchor = null;
    if (_activeOnDocClick) {
        window.removeEventListener("mousedown", _activeOnDocClick, true);
        window.removeEventListener("pointerdown", _activeOnDocClick, true);
        document.removeEventListener("mousedown", _activeOnDocClick, true);
        document.removeEventListener("pointerdown", _activeOnDocClick, true);
        _activeOnDocClick = null;
    }
}

/**
 * Build the entire nested dropdown tree synchronously, hover to expand
 * submenus, click leaf to pick. Mirrors the original ttNdropdown.js behavior.
 */
function _showDropdown(inputEl, tree, onPick) {
    _injectStyle();
    _removeDropdown();

    const root = document.createElement("ul");
    root.className = "xyplot-universal-dropdown";

    const buildLevel = (container, dict, pathParts) => {
        Object.keys(dict).forEach((key) => {
            const child = dict[key];
            const li = document.createElement("li");

            // Track hover: clear .selected on siblings and add to this one,
            // so CSS `li.selected > ul` opens the nested submenu.
            li.addEventListener("mouseover", () => {
                Array.from(container.children).forEach((s) => s.classList.remove("selected"));
                li.classList.add("selected");
            });

            const isLeaf = (child === null || typeof child !== "object");
            if (isLeaf) {
                li.classList.add("item");
                li.textContent = key;
                li.addEventListener("mousedown", (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();
                    onPick([...pathParts, key], child);
                    _removeDropdown();
                });
            } else {
                li.classList.add("folder");
                li.textContent = key;
                const nested = document.createElement("ul");
                li.appendChild(nested);
                buildLevel(nested, child, [...pathParts, key]);
            }
            container.appendChild(li);
        });
    };

    buildLevel(root, tree, []);

    // Anchor below the textarea (top-left, matching original look).
    const rect = inputEl.getBoundingClientRect();
    root.style.left = (rect.left + window.scrollX) + "px";
    root.style.top = (rect.bottom + window.scrollY - 10) + "px";

    document.body.appendChild(root);
    _activeDropdown = root;
    _activeAnchor = inputEl;

    // Dismiss when clicking outside both the dropdown and the textarea.
    // We listen on window in capture phase + on pointerdown because the
    // ComfyUI/Litegraph canvas consumes pointer events and the regular
    // bubbling `mousedown` may never reach document otherwise.
    const onDocClick = (ev) => {
        const t = ev.target;
        if (root.contains(t)) return;
        if (t === inputEl) return;
        _removeDropdown();
    };
    _activeOnDocClick = onDocClick;
    // Defer attach so the click that opened us doesn't immediately close us.
    setTimeout(() => {
        window.addEventListener("mousedown", onDocClick, true);
        window.addEventListener("pointerdown", onDocClick, true);
    }, 0);
}

/* --------------------------------------------------------------- */
/* Inspect upstream graph                                          */
/* --------------------------------------------------------------- */
function _collectUpstreamIds(node) {
    const seen = new Set();
    const stack = [node.id];
    const out = [];
    while (stack.length) {
        const id = stack.pop();
        const n = node.graph?._nodes_by_id?.[id];
        if (!n) continue;
        if (n.inputs) {
            for (const inp of n.inputs) {
                if (inp.link != null) {
                    const link = node.graph.links[inp.link];
                    if (!link) continue;
                    const src = link.origin_id;
                    if (!seen.has(src)) {
                        seen.add(src);
                        out.push(src);
                        stack.push(src);
                    }
                }
            }
        }
    }
    return out;
}

function _widgetsFor(n) {
    if (!n.widgets) return null;
    const out = {};
    for (const w of n.widgets) {
        if (!w.type) continue;
        if (WIDGETS_TO_IGNORE.has(w.name)) continue;
        if (w.type === "button") continue;
        if (typeof w.type === "string" && w.type.startsWith("converted")) continue;

        if (w.name === "seed" || w.name === "noise_seed") {
            out[w.name] = { "Random Seed": "__RANDOM_SEED__" };
            continue;
        }
        if (w.type === "toggle") {
            out[w.name] = { "True": "true", "False": "false" };
            continue;
        }
        if (["customtext", "text", "string"].includes(w.type)) {
            out[w.name] = { "(string)": "" };
            continue;
        }
        if (w.type === "number") {
            out[w.name] = { [String(w.value)]: String(w.value) };
            continue;
        }
        // combo
        if (w.options?.values) {
            const values = typeof w.options.values === "function"
                ? w.options.values() : w.options.values;
            const sub = {};
            for (const v of values) sub[String(v)] = String(v);
            out[w.name] = sub;
            continue;
        }
    }
    return Object.keys(out).length ? out : null;
}

function _buildTree(node) {
    const tree = {
        "Add Plot Line": {
            "Values only label":     "__ADDLINE_v_label__",
            "Title and values":      "__ADDLINE_tv_label__",
            "ID, title and values":  "__ADDLINE_idtv_label__",
        },
    };
    const ids = _collectUpstreamIds(node);
    for (const id of ids) {
        const n = node.graph._nodes_by_id[id];
        if (!n) continue;
        const ws = _widgetsFor(n);
        if (!ws) continue;
        const title = n.title || n.constructor?.title || n.type;
        tree[`[${id}] ${title}`] = ws;
    }
    if (Object.keys(tree).length === 1) {
        tree["(no upstream widgets found)"] = null;
    }
    return tree;
}

/* --------------------------------------------------------------- */
/* Insertion logic                                                 */
/* --------------------------------------------------------------- */
function _findNextAxisNumber(text) {
    const matches = [...text.matchAll(/<\s*(\d+)\s*:/g)];
    if (!matches.length) return 1;
    return Math.max(...matches.map(m => parseInt(m[1], 10))) + 1;
}

function _insertAtCursor(widget, value) {
    // Always read the LIVE element from the widget - ComfyUI may have
    // recreated the textarea since the dropdown was opened.
    const textarea = widget.inputEl;
    if (!textarea) return;
    const start = textarea.selectionStart ?? textarea.value.length;
    const end = textarea.selectionEnd ?? textarea.value.length;
    const before = textarea.value.substring(0, start);
    const after = textarea.value.substring(end);
    const sep = (before.length === 0 || before.endsWith("\n")) ? "" : "\n";
    const newValue = before + sep + value + after;
    textarea.value = newValue;
    const newPos = (before + sep + value).length;
    textarea.selectionStart = textarea.selectionEnd = newPos;
    // Propagate to the widget object (ComfyUI persists from `widget.value`
    // when the workflow is saved/queued). We deliberately do NOT dispatch
    // a synthetic `input` event - ComfyUI's own input listener might react
    // by re-rendering and would also re-trigger our open-on-input handler.
    widget.value = newValue;
    if (typeof widget.callback === "function") {
        // Many widgets register a callback for value changes. Mirror the
        // litegraph contract by invoking it directly.
        try { widget.callback(newValue); } catch (_) { /* ignore */ }
    }
}

function _onPick(widget, pathParts, leafValue) {
    const textarea = widget.inputEl;
    if (!textarea) return;
    const text = textarea.value;

    if (pathParts[0] === "Add Plot Line") {
        let label = "v_label";
        if (leafValue === "__ADDLINE_tv_label__") label = "tv_label";
        else if (leafValue === "__ADDLINE_idtv_label__") label = "idtv_label";
        else if (leafValue === "__ADDLINE_v_label__") label = "v_label";
        const next = _findNextAxisNumber(text);
        const out = `<${next}:${label}>\n`;
        _insertAtCursor(widget, out);
        return;
    }
    if (pathParts[0].startsWith("(")) return;

    const headMatch = pathParts[0].match(/^\[(\d+)\]/);
    if (!headMatch) return;
    const nodeId = headMatch[1];
    const widgetName = pathParts[1];
    let value = leafValue == null ? pathParts[2] : leafValue;
    if (value === "__RANDOM_SEED__") value = String(Math.floor(Math.random() * 1e15));

    let line = `[${nodeId}:${widgetName}='${value}']`;
    if (text.trim() === "") {
        line = `<1:v_label>\n${line}`;
    }
    _insertAtCursor(widget, line);
}

/* --------------------------------------------------------------- */
/* Wire up textareas                                               */
/*                                                                 */
/* ComfyUI sometimes re-creates the underlying <textarea> for      */
/* multiline widgets (resize / reconvert / hot-reload). We avoid   */
/* the resulting "stale element" bugs by:                          */
/*   - tagging the node (not the textarea) as wired, and           */
/*   - delegating events at the document level, looking up the     */
/*     owning node + widget on every event.                        */
/* --------------------------------------------------------------- */
const PLOT_WIDGET_NAMES = new Set(["x_plot", "y_plot", "z_plot"]);

// Suppression flag is keyed off the widget object's identity so it
// survives <textarea> re-creation.
const _suppressOpenFor = new WeakSet();

function _findOwningWidget(target) {
    if (!target || target.tagName !== "TEXTAREA") return null;
    if (!app.graph || !app.graph._nodes) return null;
    for (const node of app.graph._nodes) {
        if (!node || node.type !== NODE_CLASS) continue;
        if (!node.widgets) continue;
        for (const w of node.widgets) {
            if (!PLOT_WIDGET_NAMES.has(w.name)) continue;
            if (w.inputEl === target) return { node, widget: w };
        }
    }
    return null;
}

function _openDropdownFor(node, widget) {
    if (_suppressOpenFor.has(widget)) return;
    if (!widget.inputEl) return;
    const tree = _buildTree(node);
    _showDropdown(widget.inputEl, tree, (parts, leaf) => {
        // Always read the current live element from the widget at pick time.
        _onPick(widget, parts, leaf);
    });
}

let _delegationInstalled = false;
function _installDelegation() {
    if (_delegationInstalled) return;
    _delegationInstalled = true;

    // Open on a real user click on the textarea. We use `mouseup` because it
    // fires after the textarea has taken focus, matching the original
    // tinyterra behavior. We require the event target to currently be the
    // focused element so synthetic / re-entrant events can never trigger us.
    const userOpen = (ev) => {
        const owner = _findOwningWidget(ev.target);
        if (!owner) return;
        if (document.activeElement !== ev.target) return;
        _openDropdownFor(owner.node, owner.widget);
    };
    document.addEventListener("mouseup", userOpen, true);

    // Open on real keyboard input. `keyup` only fires from actual user typing
    // (synthetic `input` events ComfyUI dispatches during workflow updates do
    // NOT produce keyup events), which prevents the auto-popup glitch where
    // a value-pick would immediately reopen the picker on another textbox.
    document.addEventListener("keyup", (ev) => {
        const owner = _findOwningWidget(ev.target);
        if (!owner) return;
        if (document.activeElement !== ev.target) return;
        // Don't reopen on Escape / arrow keys / Tab / Enter etc.
        if (ev.key && ev.key.length > 1 && ev.key !== "Backspace" && ev.key !== "Delete") {
            return;
        }
        _openDropdownFor(owner.node, owner.widget);
    }, true);

    document.addEventListener("contextmenu", (ev) => {
        const owner = _findOwningWidget(ev.target);
        if (!owner) return;
        ev.preventDefault();
        ev.stopPropagation();
        _openDropdownFor(owner.node, owner.widget);
    }, true);

    document.addEventListener("focusout", (ev) => {
        const owner = _findOwningWidget(ev.target);
        if (!owner) return;
        // Defer so a click on a dropdown <li> can fire its own pick first.
        setTimeout(() => {
            if (!_activeDropdown) return;
            if (_activeAnchor !== owner.widget.inputEl) return;
            const ae = document.activeElement;
            if (_activeDropdown.contains(ae)) return;
            _removeDropdown();
        }, 100);
    }, true);
}

/* --------------------------------------------------------------- */
/* 50/50 layout: textboxes left, preview right, toggles full-width */
/* --------------------------------------------------------------- */
const _nodeObservers = new WeakMap();

function _findDirectChild(inputEl, nodeEl) {
    let el = inputEl;
    while (el.parentElement && el.parentElement !== nodeEl) {
        el = el.parentElement;
    }
    return el.parentElement === nodeEl ? el : null;
}

function _tagPreviewContainers(nodeEl) {
    const previews = nodeEl.querySelectorAll('img, canvas, [class*="preview"], [class*="output"]');
    for (const el of previews) {
        const container = _findDirectChild(el, nodeEl);
        if (container && !container.dataset.xyplotRole) {
            container.dataset.xyplotRole = 'right';
        }
    }
}

function _applyNodeLayout(node) {
    const nodeEl = document.querySelector(`[data-id="${node.id}"]`);
    if (!nodeEl) return;
    nodeEl.classList.add('xyplot-grid');

    for (const child of nodeEl.children) {
        child.dataset.xyplotRole = 'full';
    }

    for (const w of node.widgets) {
        if (!w || !w.inputEl) continue;
        const container = _findDirectChild(w.inputEl, nodeEl);
        if (container && PLOT_WIDGET_NAMES.has(w.name)) {
            container.dataset.xyplotRole = 'left';
        }
    }

    _tagPreviewContainers(nodeEl);
}

function _setupLayoutObserver(node) {
    setTimeout(() => {
        _applyNodeLayout(node);
        const nodeEl = document.querySelector(`[data-id="${node.id}"]`);
        if (!nodeEl) return;
        if (_nodeObservers.has(node)) {
            _nodeObservers.get(node).disconnect();
        }
        const observer = new MutationObserver(() => _applyNodeLayout(node));
        observer.observe(nodeEl, { childList: true, subtree: true });
        _nodeObservers.set(node, observer);
    }, 300);
}

function _cleanupLayoutObserver(node) {
    if (_nodeObservers.has(node)) {
        _nodeObservers.get(node).disconnect();
        _nodeObservers.delete(node);
    }
}

/* --------------------------------------------------------------- */
/* Strip legacy `passthrough` slot from old workflows              */
/* --------------------------------------------------------------- */
function _stripLegacyPassthrough(node) {
    // Inputs
    if (Array.isArray(node.inputs)) {
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            if (node.inputs[i] && node.inputs[i].name === "passthrough") {
                if (typeof node.removeInput === "function") {
                    node.removeInput(i);
                } else {
                    node.inputs.splice(i, 1);
                }
            }
        }
    }
    // Outputs
    if (Array.isArray(node.outputs)) {
        for (let i = node.outputs.length - 1; i >= 0; i--) {
            if (node.outputs[i] && node.outputs[i].name === "passthrough") {
                if (typeof node.removeOutput === "function") {
                    node.removeOutput(i);
                } else {
                    node.outputs.splice(i, 1);
                }
            }
        }
    }
}

app.registerExtension({
    name: "xyplot_universal.frontend",
    async setup() {
        // One delegated event handler set serves every plot node, regardless
        // of how many times ComfyUI re-creates the underlying <textarea>.
        _installDelegation();
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setTimeout(() => {
                _stripLegacyPassthrough(this);
                _setupLayoutObserver(this);
            }, 0);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            setTimeout(() => {
                _stripLegacyPassthrough(this);
                _applyNodeLayout(this);
            }, 0);
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            const r = onRemoved ? onRemoved.apply(this, arguments) : undefined;
            _cleanupLayoutObserver(this);
            return r;
        };
    },
});
