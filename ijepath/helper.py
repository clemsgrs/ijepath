# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import torch

import ijepath.models.vision_transformer as vit
from ijepath.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from ijepath.utils.tensors import trunc_normal_

logger = logging.getLogger("ijepath")


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        pass_index = int(checkpoint.get('pass_index', checkpoint.get('epoch', 0)))

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from pass_index {pass_index} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from pass_index {pass_index} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from pass_index {pass_index} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from pass_index {pass_index}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        pass_index = 0

    return encoder, predictor, target_encoder, opt, scaler, pass_index


def init_model(
    device,
    patch_size=16,
    architecture='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
    encoder = vit.__dict__[architecture](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    encoder_params = sum(int(p.numel()) for p in encoder.parameters())
    predictor_params = sum(int(p.numel()) for p in predictor.parameters())
    logger.info(
        f"Initialized model: architecture={architecture} patch_size={int(patch_size)} "
        f"crop_size={int(crop_size)} encoder_params={encoder_params} "
        f"predictor_params={predictor_params}"
    )
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    total_steps,
    start_lr,
    ref_lr,
    warmup,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    ipe_scale = float(ipe_scale)
    if ipe_scale <= 0:
        raise ValueError("ipe_scale must be > 0")

    warmup = float(warmup)
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    schedule_total_steps = max(total_steps, int(ipe_scale * total_steps))
    warmup_steps = int(warmup * total_steps)

    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=schedule_total_steps)
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=schedule_total_steps)
    use_cuda_amp = bool(use_bfloat16 and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler() if use_cuda_amp else None
    return optimizer, scaler, scheduler, wd_scheduler
