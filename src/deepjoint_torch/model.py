# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

from collections import defaultdict
from typing import Tuple

import lightning.pytorch as pl
import numpy as np
import torch

from loguru import logger
from segmentation_models_pytorch import Unet

from deepjoint_torch.constants import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PRETRAINED_BREAST_MODEL,
    PRETRAINED_DENSE_MODEL,
)
from deepjoint_torch.losses import batch_dice_score


class MammoDlModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)  # N / 1 / H W
        # we need to keep 'in_channels=3' to load weights from MammoFL team
        self.breast_unet = Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
        self.dense_unet = Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )

        status = self.breast_unet.load_state_dict(torch.load(PRETRAINED_BREAST_MODEL, map_location="cpu"))
        logger.info(f"Load PRETRAINED_BREAST_MODEL : {status = }")
        status = self.dense_unet.load_state_dict(torch.load(PRETRAINED_DENSE_MODEL, map_location="cpu"))
        logger.info(f"Load PRETRAINED_DENSE_MODEL : {status = }")

        # Set False to use multiple optimizers and custom 'training_step'
        self.automatic_optimization = False

        # to collect validation metrics
        self.validation_metrics = defaultdict(list)

    def forward_breast_unet(self, images: torch.Tensor) -> torch.Tensor:
        # transform 'images' with shape [N, 1, H, W] to [N, 3, H, W] to be compatible
        # with .breast_unet and .dense_unet
        images = images.repeat([1, 3, 1, 1])
        return self.breast_unet(images)

    def forward_dense_unet(self, images: torch.Tensor) -> torch.Tensor:
        # transform 'images' with shape [N, 1, H, W] to [N, 3, H, W] to be compatible
        # with .breast_unet and .dense_unet
        images = images.repeat([1, 3, 1, 1])
        return self.dense_unet(images)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # torch.round() : push prediction from sigmoid space to {0,1} values
        pred_breast_masks = torch.round(self.forward_breast_unet(images))
        pred_dense_masks = torch.round(self.forward_dense_unet(images * pred_breast_masks))
        return pred_breast_masks, pred_dense_masks

    @torch.no_grad()
    def predict_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx, dataloader_idx=0
    ) -> Tuple[torch.Tensor, ...]:
        image_uids, images, true_breast_masks, true_dense_masks = batch
        pred_breast_sigmoid = self.forward_breast_unet(images)  # values in range [0, 1]
        pred_dense_sigmoid = self.forward_dense_unet(
            images * torch.round(pred_breast_sigmoid)
        )  # values in range [0, 1]
        return pred_breast_sigmoid, pred_dense_sigmoid

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        breast_opt, dense_opt = self.optimizers()
        image_uids, images, true_breast_masks, true_dense_masks = batch[0]

        # Optimize 'breast' segmentation
        # stay in sigmoid range [0,1] during the training : do not apply torch.round()
        pred_breast_sigmoid = self.forward_breast_unet(images)
        breast_loss = 1 - batch_dice_score(pred_breast_sigmoid, true_breast_masks)
        breast_opt.zero_grad()
        self.manual_backward(breast_loss)
        breast_opt.step()

        # Optimize 'dense' segmentation
        # stay in sigmoid range [0,1] during the training : do not apply torch.round()
        # and don't use 'pred_breast_masks', use 'true_breast_masks' to hide pectoral muscle.
        pred_dense_sigmoid = self.forward_dense_unet(images * true_breast_masks)
        dense_loss = 1 - batch_dice_score(pred_dense_sigmoid, true_dense_masks)
        dense_opt.zero_grad()
        self.manual_backward(dense_loss)
        dense_opt.step()

        # total loss : dense + breast losses
        loss = dense_loss + breast_loss
        self.log_dict(
            {"TRAIN/loss": loss, "TRAIN/dense_loss": dense_loss, "TRAIN/breast_loss": breast_loss},
            prog_bar=True,
        )

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int, dataloader_idx: int = 0):
        image_uids, images, true_breast_masks, true_dense_masks = batch

        # if we use 'forward' we have 'threshold-ed' masks {0,1}
        pred_breast_masks, pred_dense_masks = self.forward(images)

        breast_dice_score = batch_dice_score(pred_breast_masks, true_breast_masks)
        dense_dice_score = batch_dice_score(pred_dense_masks, true_dense_masks)

        breast_loss = 1 - breast_dice_score
        dense_loss = 1 - dense_dice_score

        # total loss : dense + breast losses
        loss = breast_loss + dense_loss
        metrics = {
            "VALID_loss": loss,
            "VALID_dense_loss": dense_loss,
            "VALID_breast_loss": breast_loss,
            "VALID_breast_dice_score": breast_dice_score,
            "VALID_dense_dice_score": dense_dice_score,
        }

        for metric_name, metric in metrics.items():
            self.validation_metrics[metric_name].append(metric.cpu().numpy())

        self.log_dict(metrics)

    def on_validation_epoch_end(self) -> None:
        for metric_name, metric_values in self.validation_metrics.items():
            logger.success(f"{metric_name} = {np.mean(metric_values):.6f}")

        del self.validation_metrics
        self.validation_metrics = defaultdict(list)

    def configure_optimizers(self):
        breast_opt = torch.optim.Adam(self.breast_unet.parameters(), lr=1e-5)
        dense_opt = torch.optim.Adam(self.dense_unet.parameters(), lr=1e-5)
        return [breast_opt, dense_opt], []
