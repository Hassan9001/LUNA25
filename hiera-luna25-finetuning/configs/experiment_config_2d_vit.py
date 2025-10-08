# Portions of this code are adapted from luna25-baseline-public
# Source: https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/experiment_config.py
# License: Apache License 2.0 (see https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/LICENSE)

# Please note that this configuration is provided as a template. The parameters should be carefully tuned to achieve optimal performance.

from pathlib import Path

from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights


class Configuration(object):
    def __init__(self, kind="k-fold") -> None:
        assert kind in ["all-data", "k-fold", "submission"], \
            f"Invalid kind: {kind}. Must be one of ['all-data', 'k-fold', 'submission']."
        self.kind = kind
        workspace_root = str(Path(__file__).resolve().parent.parent)

        # Directories
        self.WORKDIR = Path(f"/{workspace_root}/hiera-luna25-finetuning")
        self.RESOURCES = self.WORKDIR / "resources"
        # Path to the nodule blocks folder provided for the LUNA25 training.
        self.DATADIR = Path(f"/{workspace_root}/luna25_nodule_blocks")
        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path(f"/{workspace_root}/data_splits")
        # Results will be saved in the results directory
        # inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if kind != "submission":
            self.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

        # Train
        self.EXPERIMENT_NAME = "finetune-vit"  # must contain "vit"
        self.MODE = "2D"
        self.SEED = 2025
        self.NUM_WORKERS = 2
        # The effective batch size is self.MINI_BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS.
        self.MINI_BATCH_SIZE = 1024
        self.GRADIENT_ACCUMULATION_STEPS = 1
        self.PATIENCE = 20
        self.NUM_GPUS = 1
        self.NUM_FOLDS = 5

        # Data & Augmentation
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.PATCH_SIZE = [64, 128, 128]
        self.ROTATION = ((-45, 45), (-45, 45), (-45, 45))
        self.TRANSLATION = True
        self.MIXUP = {
            "ENABLE": True,
            "ALPHA": 0.8,
            "CUTMIX_ALPHA": 1.0,
            "PROB": 1.0,
            "SWITCH_PROB": 0.5,
            "LABEL_SMOOTH_VALUE": 0.1
        }

        # ViT
        self.PRETRAINED_MODEL = "vit_b_16"
        self.PRETRAINED_MODEL_CONFIGS = {
            "vit_b_16": {
                "DEFAULT": ViT_B_16_Weights.IMAGENET1K_V1.url,  # IMAGENET1K_V1
                "IMAGENET1K_SWAG_E2E_V1": ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.url,
                "IMAGENET1K_SWAG_LINEAR_V1": ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.url
            },
            "vit_b_32": {
                "DEFAULT": ViT_B_32_Weights.IMAGENET1K_V1.url  # IMAGENET1K_V1
            },
            "vit_l_16": {
                "DEFAULT": ViT_L_16_Weights.IMAGENET1K_V1.url,  # IMAGENET1K_V1
                "IMAGENET1K_SWAG_E2E_V1": ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1.url,
                "IMAGENET1K_SWAG_LINEAR_V1": ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.url
            },
            "vit_l_32": {
                "DEFAULT": ViT_L_32_Weights.IMAGENET1K_V1.url  # IMAGENET1K_V1
            },
            "vit_h_14": {
                "DEFAULT": ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.url,  # IMAGENET1K_SWAG_E2E_V1
                "IMAGENET1K_SWAG_LINEAR_V1": ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.url
            }
        }

        # Solver
        self.OPTIMIZING_METHOD = "sgd"
        self.BASE_LR = 2e-3
        self.COSINE_END_LR = 2e-6
        self.MAX_EPOCH = 100
        self.BETAS = (0.9, 0.999)
        self.WEIGHT_DECAY = 0.0
        # Unused if OPTIMIZING_METHOD is not "sgd"
        self.MOMENTUM = 0.9
        self.DAMPENING = 0.0
        self.NESTEROV = True
        # Start the warm up from BASE_LR * WARMUP_FACTOR.
        self.WARMUP_FACTOR = 0.1
        # Gradually warm up the BASE_LR over this number of epochs.
        self.WARMUP_EPOCHS = 5
        # If True, start from the peak cosine learning rate after warm up.
        self.COSINE_AFTER_WARMUP = True
        # If True, perform no weight decay on parameter with one dimension (bias term, etc).
        self.ZERO_WD_1D_PARAM = True
        # The layer-wise decay of learning rate. Set to 1. to disable.
        self.LAYER_DECAY = 1.0
        self.CLIP_GRAD_L2NORM = 1.0

        if kind == "all-data":
            self.CSV_DIR = Path(f"/{workspace_root}")
            self.CSV_DIR_TRAIN = self.CSV_DIR / "LUNA25_Public_Training_Development_Data.csv"
            self.EPOCHS = 0


config = Configuration(kind="submission")
