"""
Standalone sub-graph executor for xyplot_universal.

This is a self-contained variant of the executor used by tinyterraNodes for
its advanced xyPlot, adapted so it does NOT depend on any tinyterraNodes code.

The executor walks a (sub-)prompt and re-runs only the necessary upstream
nodes for our plot node, allowing us to substitute widget values in upstream
nodes between runs to build an XY(Z) image grid.
"""

import copy
import logging
import sys
import traceback

import nodes
import torch
import comfy.model_management

try:
    # ComfyUI exposes this helper - used to format error messages.
    from execution import full_type_name
except Exception:  # pragma: no cover - older ComfyUI versions
    def full_type_name(klass):
        module = klass.__module__
        if module == "builtins":
            return klass.__qualname__
        return module + "." + klass.__qualname__


# Class types whose execution must be intercepted (replaced with a no-op fake
# class) when re-running the upstream sub-graph, so we don't recurse forever
# through our own plotting node. Populated lazily by xy_plot_node.py.
_INTERCEPT_CLASSES = {}


def register_intercept(class_type_name, fake_class):
    """Register a class_type whose execution should be intercepted by the
    sub-executor. The fake_class will be used in place of the real class so
    that the node returns immediately without doing real work."""
    _INTERCEPT_CLASSES[class_type_name] = fake_class


# --------------------------------------------------------------------------- #
# Helpers - these mirror ComfyUI's execution helpers but use our own outputs
# dict so the main ComfyUI execution cache is left alone.
# --------------------------------------------------------------------------- #
def get_input_data(inputs, class_def, unique_id, outputs=None, prompt=None, extra_data=None):
    if outputs is None:
        outputs = {}
    if prompt is None:
        prompt = {}
    if extra_data is None:
        extra_data = {}

    valid_inputs = class_def.INPUT_TYPES()
    input_data_all = {}
    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                input_data_all[x] = (None,)
                continue
            obj = outputs[input_unique_id][output_index]
            input_data_all[x] = obj
        else:
            if (("required" in valid_inputs and x in valid_inputs["required"])
                    or ("optional" in valid_inputs and x in valid_inputs["optional"])):
                input_data_all[x] = [input_data]

    if "hidden" in valid_inputs:
        h = valid_inputs["hidden"]
        for x in h:
            if h[x] == "PROMPT":
                input_data_all[x] = [prompt]
            if h[x] == "EXTRA_PNGINFO":
                input_data_all[x] = [extra_data.get('extra_pnginfo', None)]
            if h[x] == "UNIQUE_ID":
                input_data_all[x] = [unique_id]
    return input_data_all


def map_node_over_list(obj, input_data_all, func, allow_interrupt=False):
    input_is_list = False
    if hasattr(obj, "INPUT_IS_LIST"):
        input_is_list = obj.INPUT_IS_LIST

    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max([len(x) for x in input_data_all.values()])

    def slice_dict(d, i):
        return {k: (v[i] if len(v) > i else v[-1]) for k, v in d.items()}

    results = []
    if input_is_list:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(obj, func)(**input_data_all))
    elif max_len_input == 0:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(obj, func)())
    else:
        for i in range(max_len_input):
            if allow_interrupt:
                nodes.before_node_execution()
            results.append(getattr(obj, func)(**slice_dict(input_data_all, i)))
    return results


def get_output_data(obj, input_data_all):
    results = []
    uis = []
    return_values = map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True)
    for r in return_values:
        if isinstance(r, dict):
            if 'ui' in r:
                uis.append(r['ui'])
            if 'result' in r:
                results.append(r['result'])
        else:
            results.append(r)
    output = []
    if len(results) > 0:
        first_result = results[0]
        try:
            result_len = len(first_result)
        except TypeError:
            try:
                first_result = tuple(first_result)
                results[0] = first_result
                result_len = len(first_result)
            except Exception:
                result_len = 1
                results[0] = (first_result,)

        output_is_list = [False] * result_len
        if hasattr(obj, "OUTPUT_IS_LIST"):
            output_is_list = obj.OUTPUT_IS_LIST

        for i, is_list in zip(range(len(results[0])), output_is_list):
            if is_list:
                output.append([x for o in results for x in o[i]])
            else:
                output.append([o[i] for o in results])
    ui = {}
    if len(uis) > 0:
        ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
    return output, ui


def format_value(x):
    if x is None:
        return None
    if isinstance(x, (int, float, bool, str)):
        return x
    return str(x)


def recursive_execute(prompt, outputs, current_item, extra_data, executed,
                      prompt_id, outputs_ui, object_storage):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']

    if class_type in _INTERCEPT_CLASSES:
        class_def = _INTERCEPT_CLASSES[class_type]
    else:
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

    if unique_id in outputs:
        return (True, None, None)

    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            if input_unique_id not in outputs:
                result = recursive_execute(prompt, outputs, input_unique_id, extra_data,
                                           executed, prompt_id, outputs_ui, object_storage)
                if result[0] is not True:
                    return result

    input_data_all = None
    try:
        input_data_all = get_input_data(inputs, class_def, unique_id, outputs, prompt, extra_data)

        obj = object_storage.get((unique_id, class_type), None)
        if obj is None:
            obj = class_def()
            object_storage[(unique_id, class_type)] = obj

        output_data, output_ui = get_output_data(obj, input_data_all)
        outputs[unique_id] = output_data
        if len(output_ui) > 0:
            outputs_ui[unique_id] = output_ui

    except comfy.model_management.InterruptProcessingException as iex:
        logging.info("xyplot_universal: processing interrupted")
        return (False, {"node_id": unique_id}, iex)
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            for name, ins in input_data_all.items():
                input_data_formatted[name] = [format_value(v) for v in ins]
        output_data_formatted = {
            nid: [[format_value(v) for v in lst] for lst in node_outs]
            for nid, node_outs in outputs.items()
        }
        logging.error(f"!!! xyplot_universal exception during sub-graph execution: {ex}")
        logging.error(traceback.format_exc())
        return (False, {
            "node_id": unique_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted,
            "current_outputs": output_data_formatted,
        }, ex)

    executed.add(unique_id)
    return (True, None, None)


def recursive_will_execute(prompt, outputs, current_item, memo=None):
    if memo is None:
        memo = {}
    unique_id = current_item
    if unique_id in memo:
        return memo[unique_id]

    inputs = prompt[unique_id]['inputs']
    will_execute = []
    if unique_id in outputs:
        return []
    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            if input_unique_id not in outputs:
                will_execute += recursive_will_execute(prompt, outputs, input_unique_id, memo)

    memo[unique_id] = will_execute + [unique_id]
    return memo[unique_id]


def recursive_output_delete_if_changed(prompt, old_prompt, outputs, current_item):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    if class_type in _INTERCEPT_CLASSES:
        class_def = _INTERCEPT_CLASSES[class_type]
    else:
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

    is_changed_old = ''
    is_changed = ''
    to_delete = False
    if hasattr(class_def, 'IS_CHANGED'):
        if unique_id in old_prompt and 'is_changed' in old_prompt[unique_id]:
            is_changed_old = old_prompt[unique_id]['is_changed']
        if 'is_changed' not in prompt[unique_id]:
            input_data_all = get_input_data(inputs, class_def, unique_id, outputs)
            if input_data_all is not None:
                try:
                    is_changed = map_node_over_list(class_def, input_data_all, "IS_CHANGED")
                    prompt[unique_id]['is_changed'] = is_changed
                except Exception:
                    to_delete = True
        else:
            is_changed = prompt[unique_id]['is_changed']

    if unique_id not in outputs:
        return True

    if not to_delete:
        if is_changed != is_changed_old:
            to_delete = True
        elif unique_id not in old_prompt:
            to_delete = True
        elif inputs == old_prompt[unique_id]['inputs']:
            for x in inputs:
                input_data = inputs[x]
                if isinstance(input_data, list):
                    input_unique_id = input_data[0]
                    if input_unique_id in outputs:
                        to_delete = recursive_output_delete_if_changed(
                            prompt, old_prompt, outputs, input_unique_id)
                    else:
                        to_delete = True
                    if to_delete:
                        break
        else:
            to_delete = True

    if to_delete:
        outputs.pop(unique_id, None)
    return to_delete


class SubGraphExecutor:
    """A reusable sub-graph executor with its own caches.

    One instance is created per plot run. It executes mutated copies of the
    upstream sub-prompt repeatedly, caching unchanged results between runs so
    we only rerun the parts of the graph affected by mutated widget values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.outputs = {}
        self.object_storage = {}
        self.outputs_ui = {}
        self.status_messages = []
        self.success = True
        self.old_prompt = {}

    def add_message(self, event, data, broadcast: bool):
        self.status_messages.append((event, data))

    def execute(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
        if extra_data is None:
            extra_data = {}
        if execute_outputs is None:
            execute_outputs = []

        nodes.interrupt_processing(False)
        self.status_messages = []
        self.add_message("execution_start", {"prompt_id": prompt_id}, broadcast=False)

        with torch.inference_mode():
            # Drop cached outputs whose nodes no longer exist
            for o in [k for k in self.outputs if k not in prompt]:
                self.outputs.pop(o, None)
            for o in list(self.object_storage.keys()):
                if o[0] not in prompt:
                    self.object_storage.pop(o, None)
                else:
                    if o[1] != prompt[o[0]]['class_type']:
                        self.object_storage.pop(o, None)

            for x in prompt:
                recursive_output_delete_if_changed(prompt, self.old_prompt, self.outputs, x)

            current_outputs = set(self.outputs.keys())
            for x in [k for k in self.outputs_ui if k not in current_outputs]:
                self.outputs_ui.pop(x, None)

            comfy.model_management.cleanup_models()
            executed = set()
            to_execute = [(0, nid) for nid in execute_outputs]

            while to_execute:
                memo = {}
                to_execute = sorted(
                    [(len(recursive_will_execute(prompt, self.outputs, a[-1], memo)), a[-1])
                     for a in to_execute]
                )
                output_node_id = to_execute.pop(0)[-1]
                self.success, error, ex = recursive_execute(
                    prompt, self.outputs, output_node_id, extra_data,
                    executed, prompt_id, self.outputs_ui, self.object_storage)
                if self.success is not True:
                    raise Exception(ex)

            for x in executed:
                self.old_prompt[x] = copy.deepcopy(prompt[x])

            if comfy.model_management.DISABLE_SMART_MEMORY:
                comfy.model_management.unload_all_models()
