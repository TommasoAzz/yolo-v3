from torch import Tensor
from torch import nn

from yolo.blocks.convolutional import Convolutional
from yolo.blocks.scale_prediction import ScalePrediction
from yolo.blocks.yolo_block import YOLOBlock


class Detection(YOLOBlock):
    def __init__(self, index: int, in_channels: int, out_channels: int, num_classes: int, num_anchors: int):
        super().__init__(index)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.layers = nn.Sequential()

        self.layers.add_module(self._module_name("1_convolutional", index), Convolutional(
            index,
            in_channels=self.in_channels,
            out_channels=self.in_channels // 2,
            kernel_size=1
        ))
        self.layers.add_module(self._module_name("2_convolutional", index), Convolutional(
            index,
            in_channels=self.in_channels // 2,
            out_channels=self.in_channels,
            kernel_size=3,
            padding=1
        ))
        self.layers.add_module(self._module_name("3_convolutional", index), Convolutional(
            index,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1
        ))

        self.prediction = ScalePrediction(
            index,
            in_channels=self.out_channels,
            num_anchors=self.num_anchors,
            num_classes=self.num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def loss_value(self, x: Tensor) -> Tensor:
        return self.prediction(x)

    def block_type(self) -> str:
        return "DetectionBlock"
