/*
 * Frontend helper for the XY Plot (Universal) node.
 *
 * Restored to the original repo behavior (hover-cascade nested dropdown,
 * mousedown-to-pick on leaves, position:absolute on body, focusout-based
 * dismissal), plus three additions on top:
 *   1. A "?" help button drawn in the node's title bar that opens a
 *      compact syntax cheatsheet modal.
 *   2. Silent stale-textarea recovery in _findOwningWidget /
 *      _resolveLiveTextarea so the rare "stops working after running" bug
 *      can heal itself when ComfyUI recreates the underlying <textarea>.
 *   3. The legacy keyup-to-open handler is removed (typing into the
 *      textbox no longer triggers the dropdown - that was confusing).
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

/* ---------- Help modal ---------- */
.xyplot-help-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
}
.xyplot-help-modal {
    background: #1d1d1d;
    color: #ddd;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 18px 22px;
    max-width: 720px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    font-family: system-ui;
    font-size: 0.85rem;
    line-height: 1.45;
    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.6);
}
.xyplot-help-modal h2 {
    margin: 0 0 4px 0;
    font-size: 1.05rem;
}
.xyplot-help-modal h3 {
    margin: 16px 0 6px 0;
    font-size: 0.92rem;
    color: #79c0ff;
}
.xyplot-help-modal code {
    background: #2a2a2a;
    border-radius: 3px;
    padding: 1px 5px;
    font-family: "Fira Code", Consolas, monospace;
    font-size: 0.8rem;
}
.xyplot-help-modal pre {
    background: #111;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 8px 10px;
    margin: 6px 0 10px;
    overflow-x: auto;
    font-family: "Fira Code", Consolas, monospace;
    font-size: 0.78rem;
    color: #c9d1d9;
}
.xyplot-help-modal .close-row {
    text-align: right;
    margin-top: 12px;
}
.xyplot-help-modal button {
    background: #2a2a2a;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 4px 14px;
    cursor: pointer;
    font-family: inherit;
}
.xyplot-help-modal button:hover { background: #3a3a3a; }
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
 * Translate-on-wheel: hovering a list and using the scroll wheel slides the
 * list up/down via CSS transform so off-viewport items become reachable. No
 * scrollbar, no clipping. Each <ul> tracks its own offset; stopPropagation
 * keeps parent and child lists scrolling independently.
 */
function _attachWheelScroll(ul) {
    let offset = 0;
    ul.addEventListener("wheel", (ev) => {
        // Always claim wheel events on a list - prevents the event from
        // bubbling up to a parent list and scrolling both at once.
        ev.preventDefault();
        ev.stopPropagation();
        const rect = ul.getBoundingClientRect();
        const naturalTop = rect.top - offset;
        const naturalBottom = rect.bottom - offset;
        const vh = window.innerHeight;
        const margin = 8;
        const minOffset = Math.min(0, vh - margin - naturalBottom);
        const maxOffset = Math.max(0, margin - naturalTop);
        if (minOffset === 0 && maxOffset === 0) return;
        // One wheel tick = one row. Read the actual row height from the
        // first direct child <li> so the step matches whatever font / padding
        // is in effect.
        const firstItem = ul.querySelector(":scope > li");
        const step = firstItem ? firstItem.offsetHeight : 22;
        const delta = Math.sign(ev.deltaY) * step;
        let newOffset = offset - delta;
        if (newOffset < minOffset) newOffset = minOffset;
        if (newOffset > maxOffset) newOffset = maxOffset;
        if (newOffset === offset) return;
        offset = newOffset;
        ul.style.transform = `translateY(${offset}px)`;
    }, { passive: false });
}

/**
 * Build the entire nested dropdown tree synchronously, hover to expand
 * submenus, click leaf to pick. Mirrors the original ttNdropdown.js behavior.
 */
function _showDropdown(inputEl, tree, onPick, clickX, clickY) {
    _injectStyle();
    _removeDropdown();

    const root = document.createElement("ul");
    root.className = "xyplot-universal-dropdown";

    const buildLevel = (container, dict, pathParts) => {
        Object.keys(dict).forEach((key) => {
            const child = dict[key];
            const li = document.createElement("li");

            li.addEventListener("mouseover", () => {
                Array.from(container.children).forEach((s) => s.classList.remove("selected"));
                li.classList.add("selected");
            });

            const isLeaf = (child === null || typeof child !== "object");
            if (isLeaf) {
                li.classList.add("item");
                li.textContent = key;
                // Multi-event pick: in some ComfyUI states (after running
                // + tab switch, the canvas pointer layer can swallow `click`
                // before it reaches us. We register pointerup / mouseup /
                // click and dedupe via a per-li flag - whichever event fires
                // first wins. The log records which event won, so we can see
                // which one is actually getting through.
                const doPick = (kind) => (ev) => {
                    if (li._xyplotPickFired) return;
                    li._xyplotPickFired = true;
                    setTimeout(() => { li._xyplotPickFired = false; }, 300);
                    ev.preventDefault();
                    ev.stopPropagation();
                    console.log("[xyplot] pick(" + kind + "):", key,
                        ev.shiftKey ? "(shift)" : "");
                    try {
                        onPick([...pathParts, key], child);
                    } catch (err) {
                        console.error("[xyplot] pick failed:", err);
                    }
                    if (!ev.shiftKey) {
                        _removeDropdown();
                    }
                };
                // mousedown only preserves focus + stops the outside-close
                // listener; the pick fires on the up/click events below.
                li.addEventListener("mousedown", (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();
                });
                li.addEventListener("pointerup", doPick("pointerup"));
                li.addEventListener("mouseup", doPick("mouseup"));
                li.addEventListener("click", doPick("click"));
            } else {
                li.classList.add("folder");
                li.textContent = key;
                const nested = document.createElement("ul");
                li.appendChild(nested);
                buildLevel(nested, child, [...pathParts, key]);
                _attachWheelScroll(nested);
            }
            container.appendChild(li);
        });
    };

    buildLevel(root, tree, []);
    _attachWheelScroll(root);

    // Anchor at the click position when we have it (so the list pops where
    // the cursor is, not at the bottom of a long textbox). Fall back to the
    // textbox bottom for keyboard / programmatic opens.
    if (clickX != null && clickY != null) {
        root.style.left = (clickX + window.scrollX) + "px";
        root.style.top = (clickY + window.scrollY) + "px";
    } else {
        const rect = inputEl.getBoundingClientRect();
        root.style.left = (rect.left + window.scrollX) + "px";
        root.style.top = (rect.bottom + window.scrollY - 10) + "px";
    }

    document.body.appendChild(root);
    _activeDropdown = root;
    _activeAnchor = inputEl;

    const onDocClick = (ev) => {
        const t = ev.target;
        if (root.contains(t)) return;
        if (t === inputEl) return;
        _removeDropdown();
    };
    _activeOnDocClick = onDocClick;
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

function _findLastLabelKind(text) {
    /* Read the label kind from the last <n:label> header in the textbox so
       auto-added headers match what the user is already using. Defaults to
       v_label when the box has no headers yet. */
    const matches = [...text.matchAll(/<\s*\d*\s*:\s*([^>\n]+)>/g)];
    if (!matches.length) return 'v_label';
    return matches[matches.length - 1][1].trim();
}

function _resolveLiveTextarea(widget) {
    /* widget.inputEl may point to a detached <textarea> after a run -
       ComfyUI rebuilds DOM on rerender without always updating the ref.
       Re-find the live one by placeholder hint (x/y/z_plot) or positional
       index, patch widget.inputEl, and return it. */
    const ta = widget.inputEl;
    if (ta && ta.isConnected) return ta;
    if (!app.graph || !app.graph._nodes) return ta;
    for (const node of app.graph._nodes) {
        if (!node || node.type !== NODE_CLASS) continue;
        if (!node.widgets || !node.widgets.includes(widget)) continue;
        const nodeEl = document.querySelector(`[data-id="${node.id}"]`);
        if (!nodeEl) break;
        const textareas = Array.from(nodeEl.querySelectorAll("textarea"));
        const wantPrefix = (widget.name || "")[0];
        for (const t of textareas) {
            const ph = (t.placeholder || "").toLowerCase();
            if (ph.startsWith(wantPrefix)) {
                widget.inputEl = t;
                return t;
            }
        }
        const plotWidgets = node.widgets.filter(w => PLOT_WIDGET_NAMES.has(w.name));
        const idx = plotWidgets.indexOf(widget);
        if (idx >= 0 && textareas[idx]) {
            widget.inputEl = textareas[idx];
            return textareas[idx];
        }
        break;
    }
    return ta;
}

function _insertAtCursor(widget, value) {
    const textarea = _resolveLiveTextarea(widget);
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
    widget.value = newValue;
    if (typeof widget.callback === "function") {
        try { widget.callback(newValue); } catch (_) { /* ignore */ }
    }
}

function _onPick(widget, pathParts, leafValue) {
    const textarea = _resolveLiveTextarea(widget);
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

    // Every pick prepends its own fresh <n:label> header so each click
    // becomes its own axis step. The label kind matches the last existing
    // header in the textbox (v_label by default).
    const labelKind = _findLastLabelKind(text);
    const nextNum = _findNextAxisNumber(text);
    const line = `<${nextNum}:${labelKind}>\n[${nodeId}:${widgetName}='${value}']`;
    _insertAtCursor(widget, line);
}

/* --------------------------------------------------------------- */
/* Wire up textareas                                               */
/* --------------------------------------------------------------- */
const PLOT_WIDGET_NAMES = new Set(["x_plot", "y_plot", "z_plot"]);

const _suppressOpenFor = new WeakSet();

function _findOwningWidget(target) {
    if (!target || target.tagName !== "TEXTAREA") return null;
    if (!app.graph || !app.graph._nodes) return null;
    for (const node of app.graph._nodes) {
        if (!node || node.type !== NODE_CLASS) continue;
        if (!node.widgets) continue;
        // Fast path: identity match on inputEl.
        for (const w of node.widgets) {
            if (!PLOT_WIDGET_NAMES.has(w.name)) continue;
            if (w.inputEl === target) return { node, widget: w };
        }
        // Recovery: stale w.inputEl. Find the live textarea inside this
        // node's DOM by placeholder hint (X_PLOT / Y_PLOT / Z_PLOT) or
        // positional index, then heal the reference.
        const nodeEl = document.querySelector(`[data-id="${node.id}"]`);
        if (!nodeEl || !nodeEl.contains(target)) continue;
        const ph = (target.placeholder || "").toLowerCase();
        let claimed = null;
        if (ph.startsWith("x")) claimed = "x_plot";
        else if (ph.startsWith("y")) claimed = "y_plot";
        else if (ph.startsWith("z")) claimed = "z_plot";
        if (claimed) {
            for (const w of node.widgets) {
                if (w.name === claimed) {
                    w.inputEl = target;
                    return { node, widget: w };
                }
            }
        }
        const textareas = Array.from(nodeEl.querySelectorAll("textarea"));
        const idx = textareas.indexOf(target);
        if (idx >= 0) {
            const plotWidgets = node.widgets.filter(w => PLOT_WIDGET_NAMES.has(w.name));
            if (plotWidgets[idx]) {
                plotWidgets[idx].inputEl = target;
                return { node, widget: plotWidgets[idx] };
            }
        }
    }
    return null;
}

function _openDropdownFor(node, widget, clickX, clickY) {
    if (_suppressOpenFor.has(widget)) return;
    if (!widget.inputEl) return;
    const tree = _buildTree(node);
    _showDropdown(widget.inputEl, tree, (parts, leaf) => {
        _onPick(widget, parts, leaf);
    }, clickX, clickY);
}

let _delegationInstalled = false;
function _installDelegation() {
    if (_delegationInstalled) return;
    _delegationInstalled = true;

    // Open on a real user click on the textarea. If the dropdown is already
    // open for this same textarea (the user clicked it a second time while
    // it was still focused), toggle it closed instead.
    const userOpen = (ev) => {
        const owner = _findOwningWidget(ev.target);
        if (!owner) return;
        if (document.activeElement !== ev.target) return;
        if (_activeDropdown && _activeAnchor === owner.widget.inputEl) {
            _removeDropdown();
            return;
        }
        _openDropdownFor(owner.node, owner.widget, ev.clientX, ev.clientY);
    };
    document.addEventListener("mouseup", userOpen, true);

    // Typing in the textarea while the dropdown is open dismisses it - user
    // is editing, not picking. isTrusted filters out the synthetic input
    // event ComfyUI's widget.callback fires when our pick writes a value,
    // which would otherwise close the dropdown right after each pick (and
    // break shift-click multi-select).
    document.addEventListener("input", (ev) => {
        if (ev.isTrusted === false) return;
        if (!_activeDropdown) return;
        if (!_activeAnchor) return;
        if (ev.target !== _activeAnchor) return;
        _removeDropdown();
    }, true);

    document.addEventListener("contextmenu", (ev) => {
        const owner = _findOwningWidget(ev.target);
        if (!owner) return;
        ev.preventDefault();
        ev.stopPropagation();
        _openDropdownFor(owner.node, owner.widget, ev.clientX, ev.clientY);
    }, true);

    document.addEventListener("focusout", (ev) => {
        const owner = _findOwningWidget(ev.target);
        if (!owner) return;
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
/* Help modal                                                      */
/* --------------------------------------------------------------- */
let _activeHelpModal = null;
function _closeHelpModal() {
    if (_activeHelpModal && _activeHelpModal.parentNode) {
        _activeHelpModal.parentNode.removeChild(_activeHelpModal);
    }
    _activeHelpModal = null;
}

function _showHelpModal() {
    _injectStyle();
    _closeHelpModal();
    const backdrop = document.createElement("div");
    backdrop.className = "xyplot-help-backdrop";
    backdrop.addEventListener("mousedown", (ev) => {
        if (ev.target === backdrop) _closeHelpModal();
    });
    const modal = document.createElement("div");
    modal.className = "xyplot-help-modal";
    const C = (s) => `<span style="color:#888">${s}</span>`;
    modal.innerHTML = `
        <h2>XY Plot — Cheatsheet</h2>

        <h3>Step</h3>
<pre>&lt;1:v_label&gt;             ${C("# header: ordinal + label")}
&lt;:v_label&gt;              ${C("# blank ordinal = auto-numbered")}
[5:steps='20']          ${C("# node 5, widget 'steps', value 20")}
[5:cfg='7.5']           ${C("# co-vary several widgets in one cell")}
# comment line          ${C("# full-line comments are ignored")}</pre>

        <h3>Labels</h3>
<pre>v_label                 ${C("# 20, 7.5")}
tv_label                ${C("# steps: 20, cfg: 7.5")}
idtv_label              ${C("# [5] steps: 20, [5] cfg: 7.5")}
my custom text          ${C("# anything else = literal")}</pre>

        <h3>Ranges</h3>
<pre>range(10, 40, 10)       ${C("# 10, 20, 30   (end-exclusive; step optional)")}
linspace(1.0, 10.0, 5)  ${C("# 1, 3.25, 5.5, 7.75, 10")}
{a, b, c}               ${C("# explicit list")}
*                       ${C("# all legal values of a combo widget")}
random_seed(4)          ${C("# 4 fresh seeds")}
random(1.0, 10.0, 5)    ${C("# 5 random numbers in [1, 10]")}</pre>
        ${C("// Several ranges in one step are zipped (same length, no Cartesian).")}

        <h3>Text tricks</h3>
<pre>[7:text.append=', cinematic']   ${C("# append to existing value")}
[7:text='%dog;cat%']            ${C("# search;replace existing value")}</pre>

        <div class="close-row"><button type="button">Close</button></div>
    `;
    modal.querySelector("button").addEventListener("click", _closeHelpModal);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    _activeHelpModal = backdrop;
    const onKey = (ev) => {
        if (ev.key === "Escape") {
            _closeHelpModal();
            window.removeEventListener("keydown", onKey, true);
        }
    };
    window.addEventListener("keydown", onKey, true);
}

/* --------------------------------------------------------------- */
/* Help "?" button in title bar                                    */
/* --------------------------------------------------------------- */
const HELP_RADIUS = 7;
function _helpButtonCenter(node) {
    return {
        x: node.size[0] - 18,
        y: -LiteGraph.NODE_TITLE_HEIGHT / 2,
    };
}

function _drawHelpButton(node, ctx) {
    if (node.flags && node.flags.collapsed) return;
    const c = _helpButtonCenter(node);
    ctx.save();
    ctx.fillStyle = "#444";
    ctx.strokeStyle = "#aaa";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(c.x, c.y, HELP_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#eee";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("?", c.x, c.y + 0.5);
    ctx.restore();
}

function _hitHelpButton(node, pos) {
    const c = _helpButtonCenter(node);
    const dx = pos[0] - c.x;
    const dy = pos[1] - c.y;
    return (dx * dx + dy * dy) <= (HELP_RADIUS * HELP_RADIUS + 4);
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

        // Help "?" button on the title bar.
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
            _drawHelpButton(this, ctx);
            return r;
        };

        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, pos, graphCanvas) {
            if (_hitHelpButton(this, pos)) {
                _showHelpModal();
                return true;
            }
            return onMouseDown ? onMouseDown.apply(this, arguments) : undefined;
        };
    },
});
