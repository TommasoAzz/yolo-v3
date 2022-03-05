from typing import List
from torch import nn
from torch import Tensor
import torch

from yolo.config import ArchitectureConfig
from yolo.blocks import Convolutional
from yolo.blocks import Residual
from yolo.blocks import Detection
from yolo.blocks import Upsample
from yolo.blocks import YOLOBlock


class Network(nn.Module):
    def __init__(self, configuration: ArchitectureConfig):
        super().__init__()

        self.config = configuration

        if self.config is None:
            raise KeyError("The configuration is missing key 'architecture'.")

        self.blocks = nn.ModuleList(self._create_blocks(self.config.blocks))

    def forward(self, x: Tensor) -> Tensor:
        outputs = []  # |outputs| = 3, one for each scale
        route_connections = []

        for block in self.blocks:
            block_type: str = block.block_type()
            x = block(x)

            if block_type == "DetectionBlock":
                outputs.append(block.loss_value(x))
            elif block_type == "ResidualBlock" and block.route_connection:
                route_connections.append(x)
            elif block_type == "UpsampleBlock":
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        # Even if "outputs" is a list, the runtime never complains. Good!
        return outputs

    def _create_blocks(self, blocks: List[dict]) -> List[YOLOBlock]:
        layers = []
        config_to_block = {
            "convolutional": self._create_convolutional_block,
            "residual": self._create_residual_block,
            "detection": lambda i, b: self._create_detection_block(i, b, self.config.num_classes),
            "upsample": self._create_upsample_block
        }

        for index, block_config in enumerate(blocks):
            block_type: str = block_config.get("type", "")

            if block_type == "":
                raise KeyError(f"Block {index} is missing the key 'type' or it is not recognized.")

            layers.append(config_to_block[block_type](index, block_config))

        return layers

    @staticmethod
    def _create_convolutional_block(index: int, block: dict) -> YOLOBlock:
        keys = {"in_channels", "out_channels", "kernel_size", "stride", "padding"}
        if not keys.issubset(block.keys()):
            raise KeyError(f"Block {index} is missing one of the following keys: {keys}")

        return Convolutional(
            index,
            in_channels=block.get("in_channels"),
            out_channels=block.get("out_channels"),
            kernel_size=block.get("kernel_size"),
            stride=block.get("stride"),
            padding=block.get("padding")
        )

    @staticmethod
    def _create_residual_block(index: int, block: dict) -> YOLOBlock:
        keys = {"in_channels", "repetitions"}
        if not keys.issubset(block.keys()):
            raise KeyError(f"Block {index} is missing one of the following keys: {keys}")

        return Residual(
            index,
            in_channels=block.get("in_channels"),
            repetitions=block.get("repetitions"),
            route_connection=block.get("route_connection", False)
        )

    @staticmethod
    def _create_detection_block(index: int, block: dict, num_classes: int) -> YOLOBlock:
        keys = {"in_channels", "out_channels", "num_anchors"}
        if not keys.issubset(block.keys()):
            raise KeyError(f"Block {index} is missing one of the following keys: {keys}")

        return Detection(
            index,
            in_channels=block.get("in_channels"),
            out_channels=block.get("out_channels"),
            num_anchors=block.get("num_anchors"),
            num_classes=num_classes
        )

    @staticmethod
    def _create_upsample_block(index: int, block: dict) -> YOLOBlock:
        keys = {"scale_factor"}
        if not keys.issubset(block.keys()):
            raise KeyError(f"Block {index} is missing one of the following keys: {keys}")

        return Upsample(
            index,
            scale_factor=block.get("scale_factor")
        )
