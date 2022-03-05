from torch import nn
from torch import Tensor

from yolo.blocks.yolo_block import YOLOBlock
from yolo.blocks.convolutional import Convolutional


class ScalePrediction(YOLOBlock):
    def __init__(self, index: int, in_channels: int, num_classes: int, num_anchors: int):
        super().__init__(index)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.layers = nn.Sequential()

        self.layers.add_module(self._module_name("1_convolutional", index), Convolutional(
            index,
            in_channels=self.in_channels,
            out_channels=2 * self.in_channels,
            kernel_size=3,
            padding=1
        ))
        self.layers.add_module(self._module_name("out_convolutional", index), Convolutional(
            index,
            in_channels=2 * self.in_channels,
            out_channels=self.num_anchors * (self.num_classes + 5),
            batch_normalization=False,
            kernel_size=1
        ))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, grid_size_1, grid_size_2 = x.shape
        assert grid_size_1 == grid_size_2

        out = self.layers(x).reshape(
            batch_size,
            self.num_anchors,
            self.num_classes + 5,
            grid_size_1,
            grid_size_2
        ).permute(0, 1, 3, 4, 2)
        return out
        # dimensions of return --> Batch size x num_anchors x grid_size x grid_size x (5+num_classes)

    def block_type(self) -> str:
        return "ScalePredictionBlock"
