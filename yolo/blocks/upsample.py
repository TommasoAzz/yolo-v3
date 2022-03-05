from torch import nn
from torch import Tensor

from yolo.blocks.yolo_block import YOLOBlock


class Upsample(YOLOBlock):
    def __init__(self, index: int, scale_factor: int) -> None:
        super().__init__(index)

        self.scale_factor = scale_factor

        self.layers = nn.Sequential()

        self.layers.add_module(self._module_name("upsample", index), nn.Upsample(
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=True
        ))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def block_type(self) -> str:
        return "UpsampleBlock"
