# Portions of this code are adapted from SlowFast
# Source:
#    https://github.com/facebookresearch/SlowFast/blob/main/slowfast/utils/lr_policy.py
#    https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/optimizer.py
# License: Apache License 2.0 (see https://github.com/facebookresearch/SlowFast/blob/main/LICENSE)

import json
import math

import torch

from experiment_config import config


def get_lr_at_epoch(cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = lr_func_cosine(cur_epoch)
    # Perform warm up.
    if cur_epoch < config.WARMUP_EPOCHS:
        lr_start = config.BASE_LR * config.WARMUP_FACTOR
        lr_end = lr_func_cosine(config.WARMUP_EPOCHS)
        alpha = (lr_end - lr_start) / config.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cur_epoch (float): the number of epoch of the current training stage.
    """
    offset = config.WARMUP_EPOCHS if config.COSINE_AFTER_WARMUP else 0.0
    assert config.COSINE_END_LR < config.BASE_LR
    return (
        config.COSINE_END_LR
        + (config.BASE_LR - config.COSINE_END_LR)
        * (
            math.cos(math.pi * (cur_epoch - offset) /
                     (config.MAX_EPOCH - offset))
            + 1.0
        )
        * 0.5
    )


def get_param_groups(model):
    """
    Get parameter groups for the optimizer.
    """
    def _get_layer_decay(name):
        layer_id = None
        if name in ("cls_token", "mask_token"):
            layer_id = 0
        elif name.startswith("pos_embed"):
            layer_id = 0
        elif name.startswith("patch_embed"):
            layer_id = 0
        elif name.startswith("blocks"):
            layer_id = int(name.split(".")[1]) + 1
        else:
            layer_id = sum(
                config.PRETRAINED_MODEL_CONFIGS[config.PRETRAINED_MODEL]["stages"]) + 1
        layer_decay = config.LAYER_DECAY ** (sum(
            config.PRETRAINED_MODEL_CONFIGS[config.PRETRAINED_MODEL]["stages"]) + 1 - layer_id)
        return layer_id, layer_decay

    for m in model.modules():
        assert not isinstance(
            m, torch.nn.modules.batchnorm._NormBase
        ), "BN is not supported with layer decay"

    non_bn_parameters_count = 0
    zero_parameters_count = 0
    no_grad_parameters_count = 0
    parameter_group_names = {}
    parameter_group_vars = {}

    skip = {}
    if config.NUM_GPUS > 1:
        if hasattr(model.module, "no_weight_decay"):
            skip = model.module.no_weight_decay()
            # skip = {"module." + v for v in skip}
    else:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            group_name = "no_grad"
            no_grad_parameters_count += 1
            continue
        name = name[len("module."):] if name.startswith("module.") else name
        if name in skip or (
            (len(p.shape) == 1 or name.endswith(".bias"))
            and config.ZERO_WD_1D_PARAM
        ):
            layer_id, layer_decay = _get_layer_decay(name)
            group_name = f"layer_{layer_id}_zero"
            weight_decay = 0.0
            zero_parameters_count += 1
        else:
            layer_id, layer_decay = _get_layer_decay(name)
            group_name = f"layer_{layer_id}_non_bn"
            weight_decay = config.WEIGHT_DECAY
            non_bn_parameters_count += 1

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
                "layer_decay": layer_decay,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
                "layer_decay": layer_decay,
            }
        parameter_group_names[group_name]["params"].append(name)
        parameter_group_vars[group_name]["params"].append(p)

    # print(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
    optim_params = list(parameter_group_vars.values())

    # Check all parameters will be passed into optimizer.
    assert (len(list(model.parameters())) == non_bn_parameters_count + zero_parameters_count +
            no_grad_parameters_count), \
        f"parameter size does not match: {non_bn_parameters_count} + \
            {zero_parameters_count} + {no_grad_parameters_count} != {len(list(model.parameters()))}"
    # print(
    #     f"non bn {non_bn_parameters_count}, \
    #     zero {zero_parameters_count}, \
    #     no grad {no_grad_parameters_count}")

    return optim_params


def construct_optimizer(model):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
    """
    if config.LAYER_DECAY > 0.0 and config.LAYER_DECAY < 1.0:
        optim_params = get_param_groups(model)
    elif config.LAYER_DECAY == 1.0:
        bn_parameters = []
        non_bn_parameters = []
        zero_parameters = []
        no_grad_parameters = []
        skip = {}

        if config.NUM_GPUS > 1:
            if hasattr(model.module, "no_weight_decay"):
                skip = model.module.no_weight_decay()
        else:
            if hasattr(model, "no_weight_decay"):
                skip = model.no_weight_decay()

        for name_m, m in model.named_modules():
            is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
            for name_p, p in m.named_parameters(recurse=False):
                name = f"{name_m}.{name_p}".strip(".")
                if not p.requires_grad:
                    no_grad_parameters.append(p)
                elif is_bn:
                    bn_parameters.append(p)
                elif any(k in name for k in skip):
                    zero_parameters.append(p)
                elif config.ZERO_WD_1D_PARAM and (
                    len(p.shape) == 1 or name.endswith(".bias")
                ):
                    zero_parameters.append(p)
                else:
                    non_bn_parameters.append(p)

        optim_params = [
            {
                "params": bn_parameters,
                "weight_decay": 0.0,
                "layer_decay": 1.0,
            },
            {
                "params": non_bn_parameters,
                "weight_decay": config.WEIGHT_DECAY,
                "layer_decay": 1.0,
            },
            {
                "params": zero_parameters,
                "weight_decay": 0.0,
                "layer_decay": 1.0,
            },
        ]
        optim_params = [x for x in optim_params if len(x["params"]) != 0]

        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(non_bn_parameters) + len(
            bn_parameters
        ) + len(zero_parameters) + len(
            no_grad_parameters
        ), f"parameter size does not match: {len(non_bn_parameters)} + \
            {len(bn_parameters)} + {len(zero_parameters)} + {len(no_grad_parameters)} \
                != {len(list(model.parameters()))}"
        # print(f"bn {len(bn_parameters)}, non bn {len(non_bn_parameters)}, \
        #     zero {len(zero_parameters)}, no grad {len(no_grad_parameters)}")
    else:
        raise ValueError(
            f"Layer decay should be in (0, 1], but is {config.LAYER_DECAY}"
        )

    if config.OPTIMIZING_METHOD == "adamw":
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=config.BASE_LR,
            betas=config.BETAS,
            eps=1e-08,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZING_METHOD == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            lr=config.BASE_LR,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            dampening=config.DAMPENING,
            nesterov=config.NESTEROV,
        )
    else:
        raise NotImplementedError(
            f"Does not support {config.OPTIMIZING_METHOD} optimizer"
        )
    return optimizer


def set_lr(optimizer, new_lr):
    """
    Set the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr * param_group["layer_decay"]
