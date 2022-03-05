import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy
import torch
import yaml
from torch import nn, Tensor
from torch.optim import Optimizer

CONFIG_FOLDER = "cfg/"

ARCHITECTURE_CONFIG_FILE = "architecture"
TRAIN_CONFIG_FILE = "training"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_config_file(config: str) -> Dict[str, dict]:
    """
    Reads a YAML configuration file and returns its content.
    config must only be the file name, without the .yml extension.
    """
    path = CONFIG_FOLDER + config + ".yml"
    if not os.path.exists(path):
        raise ValueError(f"config = {config} is invalid. Choose a valid file.")

    cfg_file = open(path, 'r')
    cfg_file_content = cfg_file.read()
    cfg_file.close()

    loaded_config = yaml.load(cfg_file_content, Loader=yaml.Loader)

    return loaded_config


def seed_everything(seed=42):
    """
    Generates a seed, sets it to the environment variable PYTHONHASHSEED and uses it for all libraries.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: nn.Module, optimizer: Optimizer, output_file_name: str = "my_checkpoint.pth.tar"):
    """
    Saves the current state of the model and optimizer in file output_file_name.
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, output_file_name)


def load_checkpoint(input_file_name: str, model: nn.Module, optimizer: Optimizer, learning_rate: float = 0.0):
    """
    Loads from input_file_name a previous state for the model and optimizer.
    If learning_rate == 0.0 (which is the default value) then the optimizer has the learning_rate loaded from
    the checkpoint file. Set it higher to update it.
    ATTENTION: model and optimizer are updated.
    """
    checkpoint = torch.load(input_file_name, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if learning_rate > 0.0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate


def save_model(model: nn.Module, output_file_name: str = "my_model.pth.tar"):
    """
    Saves the current state of the model in file output_file_name.
    It resembles the output of save_checkpoint, in order to reuse its output file.
    """
    torch.save({"state_dict": model.state_dict()}, output_file_name)


def load_model(input_file_name: str, model: nn.Module):
    """
    Loads from input_file_name a previous state for the model.
    ATTENTION: model is updated.
    """
    checkpoint = torch.load(input_file_name, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])


def plot_metrics_graph(config_name: str, labels: List[str], metric: str, epochs: int, value_lists: List[List[Tensor]]):
    """
    Plots metrics graph.
    config_name is the name of the current configuration (the graph will be saved inside a folder with this name).
    labels can be training, validation, test or combination of them.
    metric can be accuracy (object, no object, ...), mAP, ...
    value_lists are the values of the metrics in each epoch for each label, in the same order as in labels.
    """
    plt.figure()
    labels_str = ''.join([lbl + '_' for lbl in labels])
    plt.title(f"{labels_str} - {metric} - Epochs {epochs}")
    plt.xlabel('Epochs')
    plt.ylabel(metric.lower())

    plt.xticks(numpy.arange(0, epochs, 5))

    if len(labels) == 1:
        for values in value_lists:
            plt.plot([v.item() for v in values])
    else:
        for i in range(len(value_lists)):
            plt.plot([v.item() for v in value_lists[i]], label=labels[i])
            plt.legend(loc="lower right")

    out_folder = os.path.join("results", config_name)
    os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, f"{labels_str}{metric.replace(' ', '_')}.png"))
    plt.clf()
