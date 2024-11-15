# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

import os

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from loguru import logger
from torch.utils.data import DataLoader

from deepjoint_torch.callbacks import EvaluationWriter, PredictionWriter
from deepjoint_torch.constants import BEST_CHECKPOINT
from deepjoint_torch.dataset import Dataset
from deepjoint_torch.model import MammoDlModule
from deepjoint_torch.h5 import load_image_meta
from deepjoint_torch.annotations import load_annotations


def infer(
    output_dir: Path,
    checkpoint: Path,
    h5_dir: Path = BEST_CHECKPOINT,
    annotations: Path | None = None,
    eval_model: bool = False,
) -> None:

    image_meta = load_image_meta(h5_dir)
    annotations_df = load_annotations(annotations) if eval_model else None

    dataset = Dataset(
        image_meta=image_meta,
        annotations_df=annotations_df,
        random_aug=False,
        cache=False,
    )
    logger.info(f"{len(dataset) = :_}")

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)  # increase values for higher perfs

    prediction_writer = PredictionWriter(output_dir=output_dir, image_meta=image_meta)
    if eval_model:
        # in case of evaluation : compute DICE scores and save a PNG files predicted + GT masks
        eval_writer = EvaluationWriter(output_dir=output_dir)
        callbacks = [prediction_writer, eval_writer]
    else:
        # dump only PD and DA
        eval_writer = None
        callbacks = [prediction_writer]

    accelerator, map_location = (
        ("gpu", None) if len(os.environ.get("CUDA_VISIBLE_DEVICES", "")) > 1 else ("cpu", torch.device("cpu"))
    )
    trainer = pl.Trainer(accelerator=accelerator, devices=1, callbacks=callbacks)

    logger.info(f"{checkpoint = }")

    weights = torch.load(checkpoint, map_location=map_location)
    model = MammoDlModule()
    model.load_state_dict(weights)

    model.eval()
    trainer.predict(model, dataloaders=[dataloader], return_predictions=False)

    if eval_model:
        # print averaged DICE score based on dumped CSV
        dice_scores_df = pd.read_csv(eval_writer.dice_score_file)
        logger.success(
            f"Found {dice_scores_df['image_uid'].nunique()} unique image_uid(s) (df.shape : {dice_scores_df.shape}"
        )
        logger.success(f"Averaged BREAST DICE SCORE = {np.mean(dice_scores_df['breast_dice_score'].values)}")
        logger.success(f"Averaged DENSE DICE SCORE = {np.mean(dice_scores_df['dense_dice_score'].values)}")
