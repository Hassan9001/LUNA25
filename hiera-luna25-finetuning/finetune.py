# Script for finetuning Hiera or vanilla ViT to classify a pulmonary nodule as benign or malignant.

# Portions of this code are adapted from luna25-baseline-public
# Source: https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/train.py
# License: Apache License 2.0 (see https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/LICENSE)

import os
import warnings

try:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
    import torchvision
    torchvision.disable_beta_transforms_warning()
except Exception as e:
    print(f"An error occurred during setup: {e}")

import argparse
from datetime import datetime
import logging
import math
import random

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from tqdm import tqdm

from dataloader import get_data_loader
from experiment_config import config
from mixup import MixUpBinary
from models.model_2d_vit import ViT2D
from models.model_hiera import Hiera2D, Hiera3D
from optimizer import construct_optimizer, get_lr_at_epoch, set_lr

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)


def make_weights_for_balanced_classes(labels):
    """
    Make sampling weights for the data samples.
    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / cnt_dict[label])
    return weights


def summarize_metrics(exp_save_root):
    summary = {
        "train_loss": [],
        "valid_loss": [],
        "best_auc": [],
        "epoch": []
    }

    for i in range(1, config.NUM_FOLDS + 1):
        metadata_path = exp_save_root.parent / \
            f"fold_{i}" / "config.npy"
        if metadata_path.exists():
            try:
                metadata = np.load(metadata_path, allow_pickle=True).item()
                for key in summary:
                    if key in metadata:
                        summary[key].append(metadata[key])
            except Exception as e:
                logging.info(
                    "Failed to load metadata from %s: %s", str(metadata_path), e)

    for metric, values in summary.items():
        if values:
            average = sum(values) / len(values)
            calculation_str = f"({'+'.join(map(str, values))})/{len(values)}={average:.5f}"
            logging.info("Avg %s: %s", metric, calculation_str)
        else:
            logging.info("No data available for %s", metric)


def finetune(train_csv_path, exp_save_root, valid_csv_path=None):
    """
    Finetune the model on the training set and validate on the validation set if available.
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.use_deterministic_algorithms(True)

    exp_save_root.mkdir(parents=True, exist_ok=True)

    logging.info("Training with %s", train_csv_path)
    train_df = pd.read_csv(train_csv_path)
    logging.info("Number of malignant training samples: %s",
                 train_df.label.sum())
    logging.info("Number of benign training samples: %s",
                 len(train_df) - train_df.label.sum())

    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(train_df))

    train_loader = get_data_loader(
        config.DATADIR,
        train_df,
        mode=config.MODE,
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.MINI_BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    valid_loader = None
    if valid_csv_path is not None and valid_csv_path.exists():
        logging.info("Validating with %s", valid_csv_path)
        valid_df = pd.read_csv(valid_csv_path)
        logging.info("Number of malignant validation samples: %s",
                     valid_df.label.sum())
        logging.info("Number of benign validation samples: %s",
                     len(valid_df) - valid_df.label.sum())

        valid_loader = get_data_loader(
            config.DATADIR,
            valid_df,
            mode=config.MODE,
            workers=config.NUM_WORKERS,
            batch_size=config.MINI_BATCH_SIZE,
            rotations=None,
            translations=None,
            size_mm=config.SIZE_MM,
            size_px=config.SIZE_PX,
        )

    device = torch.device("cuda:0")

    if config.MODE == "3D" and "hiera" in config.EXPERIMENT_NAME.lower():
        model = Hiera3D(image_size=config.SIZE_PX,
                        image_depth=config.DEPTH_PX).to(device)
    elif config.MODE == "2D" and "hiera" in config.EXPERIMENT_NAME.lower():
        model = Hiera2D(image_size=config.SIZE_PX).to(device)
    elif config.MODE == "2D" and "vit" in config.EXPERIMENT_NAME.lower():
        model = ViT2D(image_size=config.SIZE_PX).to(device)
    else:
        raise ValueError("Invalid MODE and/or EXPERIMENT_NAME.")

    logging.info("Number of parameters: %s", sum(p.numel()
                 for p in model.parameters()))
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info("%s: %s", name, param.size())
        else:
            logging.info("[Frozen] %s: %s", name, param.size())

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = optimizer = construct_optimizer(
        model.hiera) if "hiera" in config.EXPERIMENT_NAME.lower() else construct_optimizer(model)
    if config.MIXUP["ENABLE"]:
        mixup_fn = MixUpBinary(
            mixup_alpha=config.MIXUP["ALPHA"],
            cutmix_alpha=config.MIXUP["CUTMIX_ALPHA"],
            mix_prob=config.MIXUP["PROB"],
            switch_prob=config.MIXUP["SWITCH_PROB"],
            label_smoothing=config.MIXUP["LABEL_SMOOTH_VALUE"]
        )

    best_metric = -1
    best_metric_epoch = -1
    epochs = config.MAX_EPOCH
    patience = config.PATIENCE
    mini_batch_size = config.MINI_BATCH_SIZE
    gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS
    counter = 0

    for epoch in range(epochs):

        if config.kind == "all-data" and epoch == config.EPOCHS:
            logging.info("train completed")
            break

        if counter > patience:
            logging.info("Model not improving for %s epochs", patience)
            break

        logging.info("-" * 10)
        logging.info("epoch %s/%s", epoch + 1, epochs)

        # train
        model.train()
        epoch_train_loss = 0

        for cur_iter, batch_data in tqdm(enumerate(train_loader)):
            inputs, labels = batch_data["image"], batch_data["label"]
            labels = labels.float().to(device)
            if config.MIXUP["ENABLE"]:
                inputs, labels = mixup_fn(inputs, labels)
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), labels.squeeze())

            loss /= gradient_accumulation_steps
            loss.backward()
            epoch_train_loss += loss.item()

            gradient_accumulation_done = \
                (cur_iter + 1) % gradient_accumulation_steps == 0
            is_last_iter = cur_iter + 1 == len(train_loader)
            if gradient_accumulation_done or is_last_iter:
                epoch_exact = epoch + float(cur_iter) / len(train_loader)
                lr = get_lr_at_epoch(epoch_exact)
                set_lr(optimizer, lr)
                if config.CLIP_GRAD_L2NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.CLIP_GRAD_L2NORM)
                optimizer.step()
                optimizer.zero_grad()

        epoch_train_loss /= math.ceil(len(train_loader.dataset) /
                                      (mini_batch_size * gradient_accumulation_steps))
        logging.info("epoch %s average train loss: %.4f",
                     epoch + 1, epoch_train_loss)

        # validate
        if valid_csv_path is not None and valid_csv_path.exists():
            model.eval()
            epoch_valid_loss = 0

            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.float32, device=device)
                for batch_data in valid_loader:
                    inputs, labels = batch_data["image"], batch_data["label"]
                    labels = labels.float().to(device)
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    loss = loss_function(outputs.squeeze(), labels.squeeze())
                    epoch_valid_loss += loss.item()
                    y_pred = torch.cat([y_pred, outputs], dim=0)
                    y = torch.cat([y, labels], dim=0)

                epoch_valid_loss /= len(valid_loader)
                logging.info("epoch %s average valid loss: %.4f",
                             epoch + 1, epoch_valid_loss)

                y_pred = torch.sigmoid(
                    y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
                y = y.data.cpu().numpy().reshape(-1)

                fpr, tpr, _ = metrics.roc_curve(y, y_pred)
                auc_metric = metrics.auc(fpr, tpr)

                if auc_metric > best_metric:

                    counter = 0
                    best_metric = auc_metric
                    best_metric_epoch = epoch + 1

                    torch.save(
                        model.state_dict(),
                        exp_save_root / "best_metric_model.pth",
                    )

                    metadata = {
                        "train_csv": train_csv_path,
                        "valid_csv": valid_csv_path,
                        "config": config,
                        "best_auc": best_metric,
                        "epoch": best_metric_epoch,
                        "train_loss": epoch_train_loss,
                        "valid_loss": epoch_valid_loss
                    }
                    np.save(
                        exp_save_root / "config.npy",
                        metadata,
                    )

                    preds_df = pd.DataFrame({
                        "AnnotationID": valid_df["AnnotationID"],
                        "label": y.astype(int),
                        "probability": y_pred
                    })
                    preds_df.to_csv(
                        exp_save_root / "predictions.csv",
                        index=False
                    )

                    logging.info("saved new best metric model")

                logging.info(
                    "current epoch: %s current AUC: %.4f best AUC: %.4f at epoch %s",
                    epoch + 1,
                    auc_metric,
                    best_metric,
                    best_metric_epoch,
                )
            counter += 1

    if valid_csv_path is not None and valid_csv_path.exists():
        logging.info("train completed, best_metric: %.4f at epoch: %s",
                     best_metric, best_metric_epoch)
        summarize_metrics(exp_save_root)
    if not valid_csv_path:
        torch.save(
            model.state_dict(),
            exp_save_root / "best_metric_model.pth",
        )
        metadata = {
            "train_csv": train_csv_path,
            "config": config
        }
        np.save(
            exp_save_root / "config.npy",
            metadata,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Specify training mode: 'k-fold' or 'all-data'.")
    parser.add_argument("--k_fold", action="store_true",
                        help="Enable k-fold cross-validation training mode.")
    parser.add_argument("--all_data", action="store_true",
                        help="Enable training mode using all available data.")
    args = parser.parse_args()

    save_root = config.EXPERIMENT_DIR / \
        f"{config.EXPERIMENT_NAME}-{config.MODE}-{datetime.today().strftime('%Y%m%d')}"
    save_root.mkdir(parents=True, exist_ok=True)

    if args.k_fold and args.all_data:
        logging.info(
            "Warning: Both --k_fold and --all_data were specified. Please choose only one.")
    elif args.k_fold:
        for i in range(1, config.NUM_FOLDS + 1):
            logging.info("-" * 50)
            logging.info("Fold %s", i)
            finetune(
                train_csv_path=config.CSV_DIR / f"train_fold_{i}.csv",
                valid_csv_path=config.CSV_DIR / f"val_fold_{i}.csv",
                exp_save_root=save_root / f"fold_{i}"
            )
    elif args.all_data:
        finetune(
            train_csv_path=config.CSV_DIR_TRAIN,
            exp_save_root=save_root
        )
    else:
        logging.info(
            "No training mode specified. Please use --k_fold or --all_data.")
