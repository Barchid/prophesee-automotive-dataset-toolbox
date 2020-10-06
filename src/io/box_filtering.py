"""
Define same filtering that we apply in:
"Learning to detect objects on a 1 Megapixel Event Camera" by Etienne Perot et al.

Namely we apply 2 different filters:
1. skip all boxes before 0.5s (before we assume it is unlikely you have sufficient historic)
2. filter all boxes whose diagonal <= min_box_diag**2 and whose side <= min_box_side



Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import print_function
import numpy as np


def filter_boxes(boxes, skip_ts=int(5e5), min_box_diag=30, min_box_side=10):
    """Filters boxes according to the paper rule. 

    To note: the default represents our threshold when evaluating at VGA resolution (640*360)
    To note: we assume the initial time of the video is always 0

    Args:
        boxes (np.ndarray): structured box array with fields "ts" or "t"
    """
    ts = boxes['t'] 
    width = boxes['w']
    height = boxes['h']
    diag_square = width**2+height**2
    mask = (ts>skip_ts)*(diag_square >= min_box_diag**2)*(width >= min_box_side)*(height >= min_box_side)
    return boxes[mask]

