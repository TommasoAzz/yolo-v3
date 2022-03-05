from typing import List

import torch
from torch import Tensor

from yolo.utils.intersection_over_union import iou as iou


def nms(bboxes: List, iou_threshold: float, prob_threshold: float, box_format: str = "corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        prob_threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    # Removing all bounding boxes which have objectness score lower than the threshold.
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    # Ordering by probability (greedy approach)
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while len(bboxes) > 0:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes if _keep_bbox(chosen_box, box, iou_threshold, box_format)]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def _keep_bbox(reference_bbox: Tensor, current_bbox: Tensor, iou_threshold: float, box_format: str = "corners") -> bool:
    # Boxes (in this case, current_bbox) to be kept must have different classes from reference_bbox
    # and an IOU with the reference_bbox lower than the iou_threshold.
    prediction_difference = current_bbox[0] != reference_bbox[0]

    iou_box_chosen_box = iou(torch.tensor(reference_bbox[2:]), torch.tensor(current_bbox[2:]), box_format)

    return prediction_difference or iou_box_chosen_box < iou_threshold
