#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os.path
import warnings
from typing import List, Tuple
from datetime import datetime

import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import utils
from yolo import LossFunction, Network
from yolo.config import TrainingConfig, ArchitectureConfig
from yolo.metrics import compute_accuracy_metrics, mean_average_precision
from yolo.utils import get_evaluation_bboxes

warnings.filterwarnings("ignore")

ANCHORS = [[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]]  # TODO Load your anchors (three anchors per each of the three scales is the default)


class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            if v is not None:
                self.add_scalar(k, v)


def compute_grid_sizes_and_scaled_anchors(
        model: Network,
        architecture_cfg: ArchitectureConfig) -> Tuple[List[int], Tensor]:
    model.eval()

    with torch.no_grad():
        rand_x = torch.randn(1, architecture_cfg.color_channels, architecture_cfg.width, architecture_cfg.height)
        rand_x = rand_x.to(utils.DEVICE)
        rand_y = model(rand_x)
        grid_sizes: List[int] = [rand_y[i].shape[2] for i in range(len(rand_y))]
        scaled_anchors = torch.tensor(ANCHORS) * \
                         torch.tensor(grid_sizes).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)

        scaled_anchors = scaled_anchors.to(utils.DEVICE)

        return grid_sizes, scaled_anchors


def train(loader: DataLoader,
          model: Network,
          optimizer: Optimizer,
          loss_fn: LossFunction,
          scaler: GradScaler,
          scaled_anchors: Tensor) -> Tensor:
    model.train()  # Set up the network in "training mode"
    losses = []

    for batch_index, (x, y) in enumerate(loader):
        x = x.to(utils.DEVICE)
        y0, y1, y2 = (y[0].to(utils.DEVICE), y[1].to(utils.DEVICE), y[2].to(utils.DEVICE))

        # The next scope is required to cast float values to float16 to increase training speed
        with torch.cuda.amp.autocast():
            out = model(x)
            loss0 = loss_fn.compute(out[0], y0, scaled_anchors[0])
            loss1 = loss_fn.compute(out[1], y1, scaled_anchors[1])
            loss2 = loss_fn.compute(out[2], y2, scaled_anchors[2])
            loss = loss0 + loss1 + loss2  # loss is a Tensor of shape [1].

        # Recomputing the gradient (backward pass)
        losses.append(loss)
        optimizer.zero_grad()  # Cleans the gradient state
        scaler.scale(loss).backward()  # Scales the loss (since it is possible we go underflow)
        scaler.step(optimizer)  # Optimization run
        scaler.update()  # Scaler state update

    mean_loss = torch.mean(losses)

    return mean_loss


def compute_accuracy_and_print(
        model: Network,
        set_name: str,
        loader: DataLoader,
        train_cfg: TrainingConfig,
        compute_mAP: bool = False) -> Tuple[tuple, Tensor]:
    accuracies = compute_accuracy_metrics(model, loader, threshold=train_cfg.conf_threshold)
    print("- {} set accuracies (class, no object, object): {}% - {}% - {}%".format(
        set_name,
        round((accuracies[0] * 100).item(), 4),
        round((accuracies[1] * 100).item(), 4),
        round((accuracies[2] * 100).item(), 4)
    ))

    if compute_mAP:
        pred_boxes, true_boxes = get_evaluation_bboxes(
            loader,
            model,
            iou_threshold=train_cfg.nms_iou_threshold,
            anchors=ANCHORS,
            threshold=train_cfg.conf_threshold,
            device=utils.DEVICE
        )
        mAP = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=train_cfg.map_iou_threshold,
            box_format="midpoint",
            num_classes=10,
        )
        print(f"- mAP: {round(mAP.item(), 4)}")
        return accuracies, mAP

    return accuracies, torch.zeros(1)


def cli_argument_parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(prog="YOLOv3 - Training")

    arguments.add_argument("--training-config",
                           default="training",
                           type=str,
                           help="The name of the training configuration file to choose.")
    arguments.add_argument("--architecture-config",
                           default="architecture",
                           type=str,
                           help="The name of the architecture configuration file to choose.")

    return arguments


def main(training_config_file: str, architecture_config_file: str):
    torch.backends.cudnn.benchmark = True  # should improve performances if the input size is always equal
    torch.cuda.empty_cache()  # to clean cache of CUDA

    config_name = architecture_config_file.split('/')[-1] + '_' + training_config_file.split('/')[-1]
    tb = SummaryWriter(log_dir=os.path.join('tensorboard_logs', config_name))

    # Loading the train configuration
    train_cfg = TrainingConfig.from_dict(utils.read_config_file(training_config_file))
    train_cfg.checkpoint_file = config_name + ".pth.tar"
    print(train_cfg)

    # Loading the network configuration
    architecture_cfg = ArchitectureConfig.from_dict(utils.read_config_file(architecture_config_file))

    # Setting up the network
    model = Network(architecture_cfg).to(utils.DEVICE)
    # We want to scale ANCHORS in order to adapt to the respective grid sizes.
    # Hence, we take in input grid_sizes (list of three integers) and produce one tensor
    # with the same shape of ANCHORS.
    grid_sizes, scaled_anchors = compute_grid_sizes_and_scaled_anchors(model, architecture_cfg)

    # Setting up the optimizer, gradient scaler, loss function
    optimizer = Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay)

    loss_fn = LossFunction()

    scaler = GradScaler()

    # Generating the three datasets: training, test, evaluation and the three data loaders
    training_dataset: Dataset = None  # TODO Load your training dataset (i.e., a class inheriting from Dataset)
    training_loader = DataLoader(training_dataset)  # TODO Update with some config, if you wish

    validation_dataset: Dataset = None  # TODO Load your validation dataset (i.e., a class inheriting from Dataset)
    validation_loader = DataLoader(validation_dataset)  # TODO Update with some config, if you wish

    test_dataset: Dataset = None  # TODO Load your test dataset (i.e., a class inheriting from Dataset)
    test_loader = DataLoader(test_dataset)  # TODO Update with some config, if you wish

    # Loading previous model and optimizer states (if required)
    if train_cfg.load_model:
        utils.load_checkpoint(train_cfg.checkpoint_file, model, optimizer, train_cfg.learning_rate)

    tb.add_graph(model, next(iter(training_loader))[0].to(utils.DEVICE))

    # Storage for accuracies (if accuracy graphs are needed)
    # Training
    mean_losses = []
    training_class_prediction_accuracies = []
    training_no_object_accuracies = []
    training_object_accuracies = []
    # Validation
    validation_mAPs = []
    validation_class_prediction_accuracies = []
    validation_no_object_accuracies = []
    validation_object_accuracies = []
    # Test

    # Early stopping setup
    es_bad_epochs = train_cfg.early_stopping_bad_epochs
    es_patience = 0
    es_min_loss = np.Inf
    last_epoch = 0

    training_start_time = datetime.now()
    print(f"Beginning the training process at {training_start_time}...")
    for epoch in range(train_cfg.epoch):
        print(f"Epoch {epoch + 1}/{train_cfg.epoch}")

        # Training
        mean_loss = train(training_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        print(f"- mean loss: {round(mean_loss.item(), 8)}")
        mean_losses.append(mean_loss)

        # Accuracy computation (for training and validation)
        with torch.no_grad():
            training_accuracies, _ = compute_accuracy_and_print(model, "training", training_loader, train_cfg)
            training_class_prediction_accuracies.append(training_accuracies[0])
            training_no_object_accuracies.append(training_accuracies[1])
            training_object_accuracies.append(training_accuracies[2])

            validation_accuracies, validation_mAP = compute_accuracy_and_print(
                model, "validation", validation_loader, train_cfg, compute_mAP=True)
            validation_class_prediction_accuracies.append(validation_accuracies[0])
            validation_no_object_accuracies.append(validation_accuracies[1])
            validation_object_accuracies.append(validation_accuracies[2])
            validation_mAPs.append(validation_mAP)

        # Adding metrics on Tensorboard
        tb.add_scalar("Class prediction accuracy (training)", training_accuracies[0], epoch)
        tb.add_scalar("No object accuracy (training)", training_accuracies[1], epoch)
        tb.add_scalar("Object accuracy (training)", training_accuracies[2], epoch)
        tb.add_scalar("Loss (training)", mean_loss, epoch)

        tb.add_scalar("Class prediction accuracy (validation)", validation_accuracies[0], epoch)
        tb.add_scalar("No object accuracy (validation)", validation_accuracies[1], epoch)
        tb.add_scalar("Object accuracy (validation)", validation_accuracies[2], epoch)
        tb.add_scalar("mAP (validation)", validation_mAP, epoch)

        for name, param in model.named_parameters():
            name = name.replace('.', '/')
            tb.add_histogram(name, param.data.cpu().detach().numpy(), epoch)
            tb.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), epoch)

        # Early stopping
        if train_cfg.early_stopping:
            if epoch < 5:
                if train_cfg.save_model:
                    utils.save_checkpoint(model, optimizer, output_file_name=train_cfg.checkpoint_file)
                continue

            if epoch == 5:
                print(f"Waiting for {es_bad_epochs} consecutive epochs during which the mean loss over batches does not"
                      f" decrease...")

            if mean_loss < es_min_loss:
                # Save the model
                if train_cfg.save_model:
                    utils.save_checkpoint(model, optimizer, output_file_name=train_cfg.checkpoint_file)
                es_patience = 0
                es_min_loss = mean_loss
                last_epoch = epoch
            else:
                es_patience += 1

            if es_patience == es_bad_epochs:
                print(f"Early stopped at the {epoch + 1}-th epoch, since the mean loss over batches didn't decrease "
                      f"during the last {es_bad_epochs} epochs.")
                break
        else:
            if train_cfg.save_model:
                utils.save_checkpoint(model, optimizer, output_file_name=train_cfg.checkpoint_file)

    training_stop_time = datetime.now()
    training_total_time_minutes = (training_stop_time - training_start_time).seconds // 60
    tb.add_hparams(
        hparam_dict={**train_cfg.to_dict(), **{'training_total_time_minutes': training_total_time_minutes}},
        metric_dict={
            "Loss": None,
            "Class prediction accuracy (training)": None,
            "No object accuracy (training)": None,
            "Object accuracy (training)": None,
            "Class prediction accuracy (validation)": None,
            "No object accuracy (validation)": None,
            "Object accuracy (validation)": None,
            "mAP (validation)": None
        })
    tb.flush()
    tb.close()
    print(f"...training ended at {training_stop_time} (it took about {training_total_time_minutes} minutes).\n")

    test_start_time = datetime.now()
    print(f"Beginning the evaluation process at {test_start_time}...")
    with torch.no_grad():
        test_accuracies, test_mAP = compute_accuracy_and_print(model, "test", test_loader, train_cfg, compute_mAP=True)
        test_class_prediction_accuracy = test_accuracies[0]
        test_no_object_accuracy = test_accuracies[1]
        test_object_accuracy = test_accuracies[2]
    test_stop_time = datetime.now()
    print(f"...evaluation ended at {test_stop_time}.\n")

    print("Saving metrics graphs...")
    utils.plot_metrics_graph(config_name, ["training", "validation"],
                             "Class prediction accuracy", train_cfg.epoch,
                             [training_class_prediction_accuracies, validation_class_prediction_accuracies])
    utils.plot_metrics_graph(config_name, ["training", "validation"],
                             "Object accuracy", train_cfg.epoch,
                             [training_object_accuracies, validation_object_accuracies])
    utils.plot_metrics_graph(config_name, ["training", "validation"],
                             "No object accuracy", train_cfg.epoch,
                             [training_no_object_accuracies, validation_no_object_accuracies])
    utils.plot_metrics_graph(config_name, ["training"],
                             "Mean loss", train_cfg.epoch,
                             [mean_losses])
    utils.plot_metrics_graph(config_name, ["validation"], "mAP", train_cfg.epoch, [validation_mAPs])
    print("...saving ended.\n")

    # LaTeX table results (in order to copy them easily on the project report)
    with open(os.path.join("results", config_name, "latex.txt"), "w+") as f:
        f.write("{} & {} & {} & {} - {} - {} & {} - {} - {} & {} - {} \\\\".format(
            config_name.replace('_', '\\_'),
            last_epoch + 1,
            training_total_time_minutes,
            round(training_class_prediction_accuracies[-(1+es_patience)].item(), 2),
            round(validation_class_prediction_accuracies[-(1+es_patience)].item(), 2),
            round(test_class_prediction_accuracy.item(), 2),
            round(training_object_accuracies[-(1+es_patience)].item(), 2),
            round(validation_object_accuracies[-(1+es_patience)].item(), 2),
            round(test_object_accuracy.item(), 2),
            round(validation_mAPs[-(1+es_patience)].item(), 2),
            round(test_mAP.item(), 2)
        ))


if __name__ == "__main__":
    utils.seed_everything()

    # Setting up the argument parser.
    parser = cli_argument_parser()

    # Loading the command line arguments.
    args = parser.parse_args()

    # Training the network, validating it, testing it, saving results on Tensorboard and graph files.
    main(args.training_config, args.architecture_config)
