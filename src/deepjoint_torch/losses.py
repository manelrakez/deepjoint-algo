# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

import torch


def batch_dice_score(pred_masks: torch.Tensor, true_masks: torch.Tensor) -> torch.Tensor:
    """Takes batch of pred & true masks. Return averaged DICE score over the batch.
    It works with non-threshold-ed predictions, when torch.round() is not called"""
    score = per_sample_dice_score(pred_masks, true_masks)
    return torch.mean(score)


def per_sample_dice_score(pred_masks: torch.Tensor, true_masks: torch.Tensor) -> torch.Tensor:
    """Takes batch of pred & true masks. Return DICE score for each item in the batch.
    It works with non-threshold-ed predictions, when torch.round() is not called"""
    batch_size = pred_masks.size(0)
    smooth = 1e-7
    pred_flat = pred_masks.reshape(batch_size, -1)
    true_flat = true_masks.reshape(batch_size, -1)
    intersection = torch.sum((pred_flat * true_flat), dim=1)
    score = (2.0 * intersection + smooth) / (
        torch.sum(pred_flat, dim=1) + torch.sum(true_flat, dim=1) + smooth
    )
    return score
