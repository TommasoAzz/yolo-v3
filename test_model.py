import torch
from yolo import Network
from yolo.config import ArchitectureConfig

import utils

if __name__ == "__main__":
    cfg = ArchitectureConfig.from_dict(utils.read_config_file(utils.ARCHITECTURE_CONFIG_FILE))
    model = Network(cfg)
    assert cfg.width == cfg.height
    image_size = cfg.width
    batch_size = 1

    x = torch.randn((batch_size, cfg.color_channels, cfg.width, cfg.height))
    out = model(x)
    assert out[0].shape == (batch_size, 3, image_size // 32, image_size // 32, cfg.num_classes + 5)
    assert out[1].shape == (batch_size, 3, image_size // 16, image_size // 16, cfg.num_classes + 5)
    assert out[2].shape == (batch_size, 3, image_size // 8, image_size // 8, cfg.num_classes + 5)
    print("Success!")
