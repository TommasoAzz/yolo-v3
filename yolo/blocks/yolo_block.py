from torch import nn


class YOLOBlock(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()

        self.index = index

    def block_type(self) -> str:
        return "YOLOBlock"

    def _module_name(self, module: str, index: int) -> str:
        return f"{self.block_type()}_{module}_{index}"