# Utility functions for model handling in Hiera3D

# Portions of this code are adapted from PyTorch TorchVision (Copyright (c) Soumith Chintala 2016, all rights reserved).
# Source: https://github.com/pytorch/vision
# License: BSD 3-Clause License (see https://github.com/pytorch/vision/blob/main/LICENSE)

from collections import OrderedDict
import math
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

try:
    from experiment_config import config
except ImportError:
    import sys

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    from experiment_config import config


def hack_embeddings_spatial(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    pos_embedding_key: str = "pos_embed_spatial",
) -> "OrderedDict[str, torch.Tensor]":
    """
    Attempt to fix the bug in original Hiera.

    Bug: https://arxiv.org/pdf/2311.05613

    Adapted from:
    https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/backbones/hieradet.py

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pretrained Hiera-L.
        pos_embedding_key (str): Key of the position embedding in the state dict.
                                 Default: pos_embed_spatial.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    pos_embedding = model_state[pos_embedding_key]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(
            f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2

    if new_seq_length != seq_length:
        new_seq_length_1d = image_size // patch_size

        if not config.HACK["NPY_DIR_POS_EMBED"].exists() and not config.HACK["NPY_DIR_POS_EMBED_WINDOW"].exists():
            pretrained_model_strs = config.PRETRAINED_MODEL.split("_")
            model_name = "_".join(pretrained_model_strs[1:-1])
            model_url = None
            for k, v in config.HACK.items():
                head, sep, tail = k.partition("_")
                if sep and model_name == tail:
                    model_url = v["checkpoint"]
            if model_url is None:
                raise ValueError(
                    f"{config.PRETRAINED_MODEL} is not supported in HACK.")
            state_dict = get_pretrained_model(model_url)
            state_dict = state_dict["model"]
            # for k in state_dict.keys():
            #     if "image_encoder" in k:
            #         print(k)
            pos_embed = state_dict["image_encoder.trunk.pos_embed"]
            pos_embed_window = state_dict["image_encoder.trunk.pos_embed_window"]
            np.save(config.HACK["NPY_DIR_POS_EMBED"],
                    pos_embed.detach().cpu().numpy())
            np.save(config.HACK["NPY_DIR_POS_EMBED_WINDOW"],
                    pos_embed_window.detach().cpu().numpy())

        pos_embed = torch.from_numpy(np.load(config.HACK["NPY_DIR_POS_EMBED"]))
        pos_embed_window = torch.from_numpy(
            np.load(config.HACK["NPY_DIR_POS_EMBED_WINDOW"]))

        pos_embed = nn.functional.interpolate(pos_embed, size=(
            new_seq_length_1d, new_seq_length_1d), mode="bicubic")
        pos_embed = pos_embed + \
            pos_embed_window.tile(
                [x // y for x, y in zip(pos_embed.shape, pos_embed_window.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)

        pos_embed = pos_embed.reshape(n, new_seq_length, hidden_dim)
        model_state[pos_embedding_key] = pos_embed

    return model_state


def interpolate_embeddings_spatial(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    pos_embedding_key: str = "pos_embed_spatial",
    interpolation_mode: str = "bicubic"
) -> "OrderedDict[str, torch.Tensor]":
    """
    Interpolate positional embeddings for pretrained models.

    This is particularly useful when loading a checkpoint and applying the model
    to images with a different input resolution than the one it was trained on.

    Adapted from:
    https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pretrained model.
        pos_embedding_key (str): Key of the position embedding in the state dict.
                                 Default: pos_embed_spatial.
        interpolation_mode (str): The algorithm used for interpolation. Default: bicubic.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state[pos_embedding_key]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(
            f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding = pos_embedding.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got \
                    seq_length_1d * seq_length_1d = \
                        {seq_length_1d * seq_length_1d} and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding = pos_embedding.reshape(
            1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding = nn.functional.interpolate(
            pos_embedding,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding = new_pos_embedding.reshape(
            1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding = new_pos_embedding.permute(0, 2, 1)

        model_state[pos_embedding_key] = new_pos_embedding

    return model_state


def interpolate_embeddings_temporal(
    image_depth: int,
    patch_depth: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    pos_embedding_key: str = "pos_embed_temporal",
    interpolation_mode: str = "linear"
) -> "OrderedDict[str, torch.Tensor]":
    """
    Interpolate positional embeddings for the temporal dimension during checkpoint loading.

    Similar to `interpolate_embeddings_spatial`, this function enables applying
    pretrained models to volumes with different depths (temporal resolutions).

    Args:
        image_depth (int): Image depth of the new model.
        patch_depth (int): Patch depth of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pretrained model.
        pos_embedding_key (str): Key of the position embedding in the state dict.
                                 Default: pos_embed_temporal.
        interpolation_mode (str): The algorithm used for interpolation. Default: linear.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state[pos_embedding_key]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(
            f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = image_depth // patch_depth

    # Need to interpolate the weights for the position embedding.
    if new_seq_length != seq_length:
        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding = pos_embedding.permute(0, 2, 1)

        # Perform interpolation.
        # (1, hidden_dim, seq_length) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding = nn.functional.interpolate(
            pos_embedding,
            size=new_seq_length,
            mode=interpolation_mode,
            align_corners=True
        )

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding = new_pos_embedding.permute(0, 2, 1)

        model_state[pos_embedding_key] = new_pos_embedding

    return model_state


def download_file_with_progress(url, destination_path: Path):
    """
    Download a file from a URL to a destination Path with a progress bar.
    """
    print(f"Downloading from {url} to {destination_path}...")
    with urllib.request.urlopen(url) as response, destination_path.open("wb") as out_file:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192  # 8KB chunks
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading", leave=True) as pbar:
            for chunk in iter(lambda: response.read(block_size), b""):
                out_file.write(chunk)
                pbar.update(len(chunk))
    print("Download complete!")


def get_model_url():
    """
    Get the URL for the pretrained model based on the configuration.
    """
    pretrained_model_config = config.PRETRAINED_MODEL_CONFIGS[config.PRETRAINED_MODEL]
    if config.MODE == "3D" and "hiera" in config.EXPERIMENT_NAME.lower():
        pretrained_model_url = pretrained_model_config["mae_k400"]
    elif config.MODE == "2D" and "hiera" in config.EXPERIMENT_NAME.lower():
        pretrained_model_url = pretrained_model_config["mae_in1k"]
    elif config.MODE == "2D" and "vit" in config.EXPERIMENT_NAME.lower():
        pretrained_model_url = pretrained_model_config["DEFAULT"]
    else:
        raise ValueError(
            "Invalid pretrained model configuration. Please check MODE and EXPERIMENT_NAME.")
    return pretrained_model_url


def get_pretrained_model(pretrained_model_url=None):
    """
    Get the state dictionary for the pretrained model.
    """
    if pretrained_model_url is None:
        pretrained_model_url = get_model_url()
    config.RESOURCES.mkdir(parents=True, exist_ok=True)
    local_pretrained_model_path = config.RESOURCES / \
        Path(pretrained_model_url).name
    if not local_pretrained_model_path.exists():
        print(
            f"Local pretrained model not found at {local_pretrained_model_path}")
        download_file_with_progress(
            pretrained_model_url, local_pretrained_model_path)
    else:
        print(
            f"Local pretrained model found at {local_pretrained_model_path}. Loading directly...")
    state_dict = torch.load(local_pretrained_model_path, map_location="cpu")
    return state_dict
