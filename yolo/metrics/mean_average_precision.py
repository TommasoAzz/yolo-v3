from typing import List

import torch
from collections import Counter

from torch import Tensor

from yolo.utils.intersection_over_union import iou


def mean_average_precision(
        pred_boxes: List[Tensor],
        true_boxes: List[Tensor],
        num_classes: int,
        iou_threshold: float = 0.5,
        box_format="midpoint") -> Tensor:
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    # average_precisions = []
    average_precisions = torch.zeros(num_classes)

    for c in range(num_classes):
        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        detections = [detection for detection in pred_boxes if detection[1] == c]

        ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]

        total_true_bboxes = len(ground_truths)
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes_ctr = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # amount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        # for key, val in amount_bboxes.items():
        #    amount_bboxes[key] = torch.zeros(val)
        amount_bboxes = {key: torch.zeros(val) for key, val in amount_bboxes_ctr.items()}

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        true_positives = torch.zeros(len(detections))
        false_positives = torch.zeros(len(detections))

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0
            best_gt_idx = 0  # TODO Check if it makes sense

            for idx, gt in enumerate(ground_truth_img):
                current_iou = iou(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format
                )

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    true_positives[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    false_positives[detection_idx] = 1
            else:  # if IOU is lower than the detection is a false positive
                false_positives[detection_idx] = 1

        true_positives_sum = torch.cumsum(true_positives, dim=0)
        false_positives_sum = torch.cumsum(false_positives, dim=0)
        recalls = true_positives_sum / (total_true_bboxes + 1e-6)
        precisions = true_positives_sum / (true_positives_sum + false_positives_sum + 1e-6)
        # Steps required for integrating precision-recall graph
        # - Adding 1 to the precisions
        # - Adding 0 to the recalls
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration i.e. calculating the area under the graph
        # average_precisions.append(torch.trapz(precisions, recalls))
        average_precisions[c] = torch.trapz(precisions, recalls)

    # return sum(average_precisions) / len(average_precisions)
    return torch.mean(average_precisions)
