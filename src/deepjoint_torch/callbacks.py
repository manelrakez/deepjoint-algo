# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

from pathlib import Path
from typing import Any, Optional, Sequence

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from lightning.pytorch.callbacks import BasePredictionWriter
from loguru import logger
from matplotlib import pyplot as plt

from deepjoint_torch.constants import IMAGE_HEIGHT
from deepjoint_torch.losses import per_sample_dice_score


class PredictionWriter(BasePredictionWriter):
    debug: bool = False

    def __init__(self, output_dir: Path, image_meta: pd.DataFrame):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.image_meta = image_meta[["pixel_spacing", "image_height"]].copy()

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.prediction_file = self.output_dir / "predictions.csv"
        self.csv_columns = ["image_uid", "pred_percent_density", "pred_dense_area", "pred_breast_area"]
        pd.DataFrame(columns=self.csv_columns).to_csv(self.prediction_file, index=False)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        image_uids, images, _, _ = batch
        pred_breast_sigmoid, pred_dense_sigmoid = prediction

        # for the moment we don't keep 'sigmoid' mask : use torch to have {0,1} predictions.
        pred_breast_masks = torch.round(pred_breast_sigmoid)
        pred_dense_masks = torch.round(pred_dense_sigmoid)

        if self.debug:
            logger.debug(f"image_uids : {type(image_uids)} | {image_uids}")
            logger.debug(f"images : {type(images)} | {images.shape} | {images.dtype}")
            logger.debug(
                f"pred_breast_sigmoid : {type(pred_breast_sigmoid)} | {pred_breast_sigmoid.shape} | {pred_breast_sigmoid.dtype}"
            )
            logger.debug(
                f"pred_dense_sigmoid : {type(pred_dense_sigmoid)} | {pred_dense_sigmoid.shape} | {pred_dense_sigmoid.dtype}"
            )

            logger.debug(
                f"pred_breast_masks : {type(pred_breast_masks)} | {pred_breast_masks.shape} | {pred_breast_masks.dtype}"
            )
            logger.debug(
                f"pred_breast_masks : {type(pred_dense_masks)} | {pred_dense_masks.shape} | {pred_dense_masks.dtype}"
            )

        pred_breast_masks = pred_breast_masks.cpu().numpy()
        pred_dense_masks = pred_dense_masks.cpu().numpy()

        predictions_df = []
        for index_image, image_uid in enumerate(image_uids):
            pred_breast_mask = pred_breast_masks[index_image]
            pred_dense_mask = pred_dense_masks[index_image]

            pred_breast_tissue = np.sum(pred_breast_masks[index_image])  # nb of pixels in 'breast_mask'
            pred_dense_tissue = np.sum(pred_dense_masks[index_image])  # nb of pixel in 'dense_mask'
            pred_percent_density = (pred_dense_tissue / pred_breast_tissue) * 100

            original_pixel_spacing = self.image_meta.at[
                image_uid, "pixel_spacing"
            ]  # for example : (0.1, 0.1)
            original_image_height = self.image_meta.at[image_uid, "image_height"]  # for example : 2294
            zoom_factor = IMAGE_HEIGHT / original_image_height  # for example : 576 / 2294 ~= 0.25

            rescaled_pixel_spacing = (
                original_pixel_spacing[0] / zoom_factor,
                original_pixel_spacing[1] / zoom_factor,
            )  # for example : (0.4, 0.4)

            pixel_area = np.prod(rescaled_pixel_spacing)  # area in mm2 for 1 pixel : 0.16 mm^2
            pred_dense_area = pixel_area * pred_dense_tissue
            pred_breast_area = pixel_area * pred_breast_tissue

            predictions_df.append(
                {
                    "image_uid": image_uid,
                    "pred_percent_density": pred_percent_density,
                    "pred_dense_area": pred_dense_area,
                    "pred_breast_area": pred_breast_area,
                }
            )

            if self.debug:
                logger.debug(f"\t{image_uid = }")
                logger.debug(f"\tpred_breast_mask : {pred_breast_mask.shape} | {pred_breast_mask.dtype}")
                logger.debug(f"\tpred_dense_mask : {pred_dense_mask.shape} | {pred_dense_mask.dtype}")
                logger.debug(f"\t{original_pixel_spacing = }")
                logger.debug(f"\t{original_image_height = }")
                logger.debug(f"\t{zoom_factor = }")
                logger.debug(f"\t{rescaled_pixel_spacing = }")
                logger.debug(f"\t{pixel_area = }")
                logger.debug(f"\t{pred_percent_density = :.7f}")
                logger.debug(f"\t{pred_dense_area = :.7f}")
                logger.debug(f"\t{pred_breast_area = :.7f}")

        predictions_df = pd.DataFrame(predictions_df)
        predictions_df[self.csv_columns].to_csv(self.prediction_file, mode="a", header=False, index=False)

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        pass


class EvaluationWriter(BasePredictionWriter):
    debug: bool = False

    def __init__(self, output_dir: Path):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.image_dir = output_dir / "segmentations"
        self.image_dir.mkdir(exist_ok=True)

        self.dice_score_file = self.output_dir / "dice_scores.csv"
        self.csv_columns = ["image_uid", "breast_dice_score", "dense_dice_score"]
        pd.DataFrame(columns=self.csv_columns).to_csv(self.dice_score_file, index=False)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        image_uids, images, true_breast_masks, true_dense_masks = batch
        pred_breast_sigmoid, pred_dense_sigmoid = prediction

        # for the moment we don't keep 'sigmoid' mask : use torch to have {0,1} predictions.
        pred_breast_masks = torch.round(pred_breast_sigmoid)
        pred_dense_masks = torch.round(pred_dense_sigmoid)

        if self.debug:
            logger.debug(f"image_uids : {type(image_uids)} | {image_uids}")
            logger.debug(f"images : {type(images)} | {images.shape} | {images.dtype}")
            logger.debug(
                f"true_breast_masks : {type(true_breast_masks)} | {true_breast_masks.shape} | {true_breast_masks.dtype}"
            )
            logger.debug(
                f"true_dense_masks : {type(true_dense_masks)} | {true_dense_masks.shape} | {true_dense_masks.dtype}"
            )
            logger.debug(
                f"pred_breast_sigmoid : {type(pred_breast_sigmoid)} | {pred_breast_sigmoid.shape} | {pred_breast_sigmoid.dtype}"
            )
            logger.debug(
                f"pred_dense_sigmoid : {type(pred_dense_sigmoid)} | {pred_dense_sigmoid.shape} | {pred_dense_sigmoid.dtype}"
            )

        # compute 'DICE' score with 'rounded' predictions (
        breast_dice_scores = per_sample_dice_score(pred_breast_masks, true_breast_masks).cpu().numpy()
        dense_dice_scores = per_sample_dice_score(pred_dense_masks, true_dense_masks).cpu().numpy()

        images = images.cpu().numpy()
        true_breast_masks = true_breast_masks.cpu().numpy()
        true_dense_masks = true_dense_masks.cpu().numpy()
        pred_breast_masks = pred_breast_masks.cpu().numpy()
        pred_dense_masks = pred_dense_masks.cpu().numpy()

        dice_scores_df = []

        for index_image, image_uid in enumerate(image_uids):
            image = images[index_image]

            true_breast_mask = true_breast_masks[index_image]
            true_dense_mask = true_dense_masks[index_image]
            pred_breast_mask = pred_breast_masks[index_image]
            pred_dense_mask = pred_dense_masks[index_image]

            breast_dice_score = breast_dice_scores[index_image]
            dense_dice_score = dense_dice_scores[index_image]

            dice_scores_df.append(
                {
                    "image_uid": image_uid,
                    "breast_dice_score": breast_dice_score,
                    "dense_dice_score": dense_dice_score,
                }
            )

            if self.debug:
                logger.debug(f"\timage_uid : {image_uid}")
                logger.debug(f"\tbreast_dice_score : {breast_dice_score:.7f}")
                logger.debug(f"\tdense_dice_score : {dense_dice_score:.7f}")
                logger.debug(f"\timage : {image.shape} | {image.dtype}")
                logger.debug(f"\ttrue_breast_mask : {true_breast_mask.shape} | {true_breast_mask.dtype}")
                logger.debug(f"\ttrue_dense_mask : {true_dense_mask.shape} | {true_dense_mask.dtype}")
                logger.debug(f"\tpred_breast_mask : {pred_breast_mask.shape} | {pred_breast_mask.dtype}")
                logger.debug(f"\tpred_dense_mask : {pred_dense_mask.shape} | {pred_dense_mask.dtype}")

            fig = create_pyplot_fig(
                image_uid,
                image,
                true_breast_mask,
                true_dense_mask,
                pred_breast_mask,
                pred_dense_mask,
                breast_dice_score,
                dense_dice_score,
            )

            output_file = f"{image_uid}.png"
            fig.savefig(str(self.image_dir / output_file))
            plt.close(fig)

        dice_scores_df = pd.DataFrame(dice_scores_df)
        dice_scores_df[self.csv_columns].to_csv(self.dice_score_file, mode="a", header=False, index=False)

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        pass


def create_pyplot_fig(
    image_uid: str,
    image: np.ndarray,
    true_breast_mask: np.ndarray,
    true_dense_mask: np.ndarray,
    pred_breast_mask: np.ndarray,
    pred_dense_mask: np.ndarray,
    breast_dice_score: float,
    dense_dice_score: float,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30, 12))

    fig.suptitle(f"{image_uid = }", fontsize=16)

    img_ax = axes[0]
    img_ax.imshow(np.squeeze(image), cmap="gray")  # images are [1, H, W] arrays
    img_ax.imshow(np.squeeze(true_breast_mask), alpha=0.3)
    img_ax.imshow(np.squeeze(true_dense_mask), alpha=0.3)
    img_ax.set_title("true breast & dense masks")

    input_breast_model_ax = axes[1]
    input_breast_model_ax.imshow(np.squeeze(image), cmap="gray")  # images are [1, H, W] arrays
    input_breast_model_ax.set_title("Input image for 'breast' model")

    output_breast_model_ax = axes[2]
    output_breast_model_ax.imshow(np.squeeze(pred_breast_mask))  # masks are [1, H, W] arrays
    output_breast_model_ax.set_title(f"Output mask for 'breast' model. DICE = {breast_dice_score:.7f}")

    input_dense_model_ax = axes[3]
    input_dense_model_ax.imshow(np.squeeze(image) * np.squeeze(pred_breast_mask), cmap="gray")
    input_dense_model_ax.set_title("Input image for 'dense' model")

    output_dense_model_ax = axes[4]
    output_dense_model_ax.imshow(np.squeeze(pred_dense_mask))  # masks are [1, H, W] arrays
    output_dense_model_ax.set_title(f"Output mask for 'dense' model. DICE = {dense_dice_score:.7f}")

    return fig
