from __future__ import annotations


class ArchitectureConfig:
    def __init__(self, **kwargs):
        keys = {"width", "height", "color_channels", "num_classes", "blocks"}
        if not keys.issubset(kwargs.keys()):
            raise KeyError(f"Architecture config is missing one of the following keys: {keys}")

        self.width = int(kwargs.get("width"))
        self.height = int(kwargs.get("height"))
        self.color_channels = int(kwargs.get("color_channels"))
        self.num_classes = int(kwargs.get("num_classes"))
        self.blocks = kwargs.get("blocks")

    @staticmethod
    def from_dict(config: dict) -> ArchitectureConfig | None:
        cfg_dictionary = config.get("architecture")

        if cfg_dictionary is None:
            print("FAILURE: configuration file is not well formed.")
            return None

        return ArchitectureConfig(**cfg_dictionary)

    def __repr__(self):
        return f"""ArchitectureConfig:
            width: {self.width}
            height: {self.height}
            color_channels: {self.color_channels}
            num_classes: {self.num_classes}
            blocks: {self.blocks}"""
