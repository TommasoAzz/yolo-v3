from typing import Tuple

import torch
from torch import Tensor


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def iou(boxes_preds: Tensor, boxes_labels: Tensor, box_format: str = "midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    assert box_format == "midpoint" or box_format == "corners"

    if box_format == "midpoint":
        # Box 1
        box1_x1, box1_y1, box1_x2, box1_y2 = _iou_midpoint_box_coordinates(boxes_preds)
        # Box 2
        box2_x1, box2_y1, box2_x2, box2_y2 = _iou_midpoint_box_coordinates(boxes_labels)
    else:  # box_format == "corners":
        # Box 1
        box1_x1, box1_y1, box1_x2, box1_y2 = _iou_corners_box_coordinates(boxes_preds)
        # Box 2
        box2_x1, box2_y1, box2_x2, box2_y2 = _iou_corners_box_coordinates(boxes_labels)

    # Computing the intersection
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) ensures the intersection is 0 when boxes do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Computing the union
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection + 1e-6  # + 1e-6 to ensure there is no division by 0.

    return intersection / union


def _iou_midpoint_box_coordinates(box: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x1 = box[..., 0:1] - box[..., 2:3] / 2
    y1 = box[..., 1:2] - box[..., 3:4] / 2
    x2 = box[..., 0:1] + box[..., 2:3] / 2
    y2 = box[..., 1:2] + box[..., 3:4] / 2

    return x1, y1, x2, y2


def _iou_corners_box_coordinates(box: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x1 = box[..., 0:1]
    y1 = box[..., 1:2]
    x2 = box[..., 2:3]
    y2 = box[..., 3:4]

    return x1, y1, x2, y2
