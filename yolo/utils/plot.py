import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

import utils
from yolo.utils import cells_to_bboxes, nms


def plot_image(image: Tensor, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = []  # TODO Load your class labels
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()


def plot(img, y, scaled_anchors, is_pred):
    boxes = []
    for i in range(len(y)):
        anchors_ith_layer = scaled_anchors[i]
        converted_bboxes = cells_to_bboxes(predictions=y[i],
                                           anchors=anchors_ith_layer,
                                           S=y[i].shape[2],
                                           is_preds=is_pred)
        boxes += converted_bboxes[0]
    nms_boxes = nms(bboxes=boxes,
                    iou_threshold=0.8,
                    prob_threshold=0.7,
                    box_format="midpoint")
    plot_image(img[0].permute(1, 2, 0).to("cpu"), nms_boxes)
