"""
xyplot_universal - Drop-anywhere XY-plot node for ComfyUI.

"""

from .xy_plot_node import XYPlotUniversalNode, CLASS_TYPE


NODE_CLASS_MAPPINGS = {
    CLASS_TYPE: XYPlotUniversalNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    CLASS_TYPE: "XY Plot (Universal)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
