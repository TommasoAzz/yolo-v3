import torch
from torch import nn, Tensor

from yolo.utils import iou


class LossFunction:
    def __init__(self,
                 lambda_class: float = 1,
                 lambda_noobj: float = 10,
                 lambda_obj: float = 1,
                 lambda_box: float = 10):
        # Loss functions (to compute parts of the YOLO loss)
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box

    def compute(self, predictions: Tensor, target: Tensor, anchors: Tensor):
        # predictions and target have the first four (out of five) dimensions which are equal.
        # predictions's last dimension is: 5 + num_classes.
        # target's last dimension is: 6 (prob_obj, x, y, w, h, class_label)
        # These four are: batch_size x num_anchors x grid_size x grid_size.
        # We create two tensors obj and noobj of shape:
        # - batch_size x num_anchors x grid_size x grid_size with values in {true, false}.
        # - obj has true values when the objectness score in the target is 1, false otherwise.
        # - noobj has true values when the objectness score in the target is 0, false otherwise.
        # - obj is the opposite of noobj and viceversa.
        # Notes:
        # - "..." in the selection of tensor dimensions means "through all dimensions".
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # Computing the "no object loss"
        # ------------------------------
        no_object_loss = self.bce(predictions[..., 0:1][noobj], target[..., 0:1][noobj])  # Sigmoid + BCE

        # Computing the "object loss"
        # ---------------------------
        # anchors has shape num_anchors x 2
        # num_anchors is the number of anchors in the detection layer
        # 2 because (w, h) with w and h being the width and height of the anchor boxes
        num_anchors, _ = anchors.shape
        # this reshaping is required to utilize broadcasting during adjusting of anchors
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)

        # width_anchor_adjusted = width_anchor_scaled01 * exp_t_w and for height
        # Notes:
        # - dim=-1 on torch.cat stands for concatenating the tensor along the last dimension.
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()  # .detach() not to impact the computational graph
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious)

        # Computing the "bounding box loss"
        # ---------------------------------
        # - we first compute the sigmoid of the bbox center coordinates
        # - then we scale down target values of width and height
        # Notes:
        # - 1e-16 is summed inside the logarithm to avoid computing log(0).
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors)) # Inverse of what is performed for box_preds
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # Computing the "class loss"
        # --------------------------
        class_loss = self.entropy(predictions[..., 5:][obj], target[..., 5][obj].long())  # LogSoftmax + Negative log likelihood

        return self.lambda_box * box_loss \
               + self.lambda_obj * object_loss \
               + self.lambda_noobj * no_object_loss \
               + self.lambda_class * class_loss
