# Vision Transformer for 2D image classification

# References:
# VisionTransformer: https://docs.pytorch.org/vision/main/models/vision_transformer.html

from pathlib import Path

import torch
from torch import nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
from torchvision.models.vision_transformer import interpolate_embeddings

try:
    from experiment_config import config
except ImportError:
    import sys

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    from experiment_config import config

from .model_utils import get_pretrained_model


class ViT2D(nn.Module):
    def __init__(self, image_size, kind="pretrained", num_classes=1, patch_size=16):
        super().__init__()
        assert kind in ["pretrained",
                        "finetuned"], f"Unknown model kind: {kind}"

        if config.PRETRAINED_MODEL == "vit_b_16":
            self.vit = vit_b_16(image_size=image_size)
        elif config.PRETRAINED_MODEL == "vit_b_32":
            self.vit = vit_b_32(image_size=image_size)
        elif config.PRETRAINED_MODEL == "vit_l_16":
            self.vit = vit_l_16(image_size=image_size)
        elif config.PRETRAINED_MODEL == "vit_l_32":
            self.vit = vit_l_32(image_size=image_size)
        elif config.PRETRAINED_MODEL == "vit_h_14":
            self.vit = vit_h_14(image_size=image_size)
        else:
            raise ValueError(
                f"Unknown pretrained model: {config.PRETRAINED_MODEL}")

        if kind == "pretrained":
            state_dict = get_pretrained_model()

            new_state_dict = interpolate_embeddings(
                image_size=image_size,
                patch_size=patch_size,
                model_state=state_dict
            )
            self.vit.load_state_dict(new_state_dict, strict=True)

        self.vit.heads.head = nn.Linear(
            self.vit.heads.head.in_features, num_classes)

        # initialize head (as done in ViT)
        nn.init.zeros_(self.vit.heads.head.weight)
        nn.init.zeros_(self.vit.heads.head.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return ["pos_embedding"]

    def forward(self, x):
        return self.vit(x)


if __name__ == "__main__":

    # Example usage:
    # python -m hiera-luna25-finetuning.models.model_2d_vit

    IMG_SIZE = 64
    model = ViT2D(image_size=IMG_SIZE)
    print(
        f"Output shape: {model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE)).shape}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.size()}")
        else:
            print(f"[Frozen] {name}: {param.size()}")
