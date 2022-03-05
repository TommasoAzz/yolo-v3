from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import utils
from yolo.network import Network


def compute_accuracy_metrics(
        model: Network,
        dataset_loader: DataLoader,
        threshold: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns the accuracies (in [0,1]) computed on the model.
    They are returned with the following order: class prediction, objectness, no objectness.
    """
    model.eval()

    correct_class_predictions = torch.zeros(1).to(utils.DEVICE)
    tot_class_predictions = torch.zeros(1).to(utils.DEVICE)

    correct_noobj = torch.zeros(1).to(utils.DEVICE)
    tot_noobj = torch.zeros(1).to(utils.DEVICE)

    correct_obj = torch.zeros(1).to(utils.DEVICE)
    tot_obj = torch.zeros(1).to(utils.DEVICE)

    for idx, (x, y) in enumerate(dataset_loader):
        x = x.to(utils.DEVICE)

        with torch.no_grad():
            out = model(x)

        for i in range(len(out)):
            y[i] = y[i].to(utils.DEVICE)
            # The following obj and noobj are the 1^obj_i and 1^noobj_i functions in the YOLO loss function.
            obj = y[i][..., 0] == 1
            noobj = y[i][..., 0] == 0

            correct_class_predictions += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_predictions += torch.sum(obj)

            obj_predictions = torch.sigmoid(out[i][..., 0]) > threshold

            correct_obj += torch.sum(obj_predictions[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)

            correct_noobj += torch.sum(obj_predictions[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    class_prediction_accuracy = _accuracy_value(correct_class_predictions, tot_class_predictions)
    no_object_accuracy = _accuracy_value(correct_noobj, tot_noobj)
    object_accuracy = _accuracy_value(correct_obj, tot_obj)

    return class_prediction_accuracy, no_object_accuracy, object_accuracy


def _accuracy_value(correct: Tensor, total: Tensor) -> Tensor:
    return correct / (total + 1e-16)
