from collections import OrderedDict

from torch import nn
from torch import Tensor

from yolo.blocks.yolo_block import YOLOBlock
from yolo.blocks.convolutional import Convolutional


class Residual(YOLOBlock):
    def __init__(self, index: int, in_channels: int, repetitions: int = 1, route_connection: bool = False):
        super().__init__(index)

        self.in_channels = in_channels
        self.repetitions = repetitions
        self.route_connection = route_connection
        self.layers = nn.ModuleList()

        for i in range(repetitions):
            self.layers.append(nn.Sequential(OrderedDict([
                (self._module_name(f"r{i}_1_convolutional", index), Convolutional(
                    index,
                    in_channels=self.in_channels,
                    out_channels=self.in_channels // 2,
                    kernel_size=1
                )),
                (self._module_name(f"r{i}_2_convolutional", index), Convolutional(
                    index,
                    in_channels=self.in_channels // 2,
                    out_channels=self.in_channels,
                    kernel_size=3,
                    padding=1
                ))
            ])))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x

    def block_type(self) -> str:
        return "ResidualBlock"
