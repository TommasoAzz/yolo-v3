from torch import nn
from torch import Tensor

from yolo.blocks.yolo_block import YOLOBlock


class Convolutional(YOLOBlock):
    def __init__(self, index: int, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1, batch_normalization: bool = True):
        super().__init__(index)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.batch_normalization = batch_normalization
        self.layers = nn.Sequential()

        self.layers.add_module(self._module_name("convolutional", index), nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            bias=(not self.batch_normalization),
            stride=self.stride,
            kernel_size=self.kernel_size,
            padding=self.padding
        ))
        if self.batch_normalization:
            self.layers.add_module(self._module_name("batch_normalization", index), nn.BatchNorm2d(
                self.out_channels
            ))
            self.layers.add_module(self._module_name("leaky_relu", index), nn.LeakyReLU(0.1))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def block_type(self) -> str:
        return "ConvolutionalBlock"
