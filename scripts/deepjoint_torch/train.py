import sys, os, glob

from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from loguru import logger
from torch.utils.data import DataLoader

import pandas as pd
import re
import shutil
import torch

from deepjoint_torch.log import set_logger
from deepjoint_torch.dataset import Dataset
from deepjoint_torch.model import MammoDlModule
from deepjoint_torch.h5 import load_image_meta
from deepjoint_torch.annotations import load_annotations

from sklearn.model_selection import KFold

K_FOLDS = 10


def train(output_dir: Path, h5_dir: Path, annotations: Path) -> None:
    # Load image metadata
    image_meta = load_image_meta(h5_dir)

    # Load annotations
    annotations_df = load_annotations(annotations)
    image_meta = image_meta[image_meta["image_uid"].isin(annotations_df["image_uid"].values)]

    # Split data
    patient_metadata = image_meta.drop_duplicates(subset="patient_id")[
        ["patient_id", "source", "harmonized_manufacturer", "breast_density_score", "view"]
    ]
    grouped = patient_metadata.groupby(["breast_density_score", "harmonized_manufacturer", "source", "view"])

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for group, data in grouped:
        train = data.sample(frac=0.9, random_state=748)
        test = data.drop(train.index)
        train_data = pd.concat([train_data, train])
        test_data = pd.concat([test_data, test])

    train_data.drop(
        columns=["source", "harmonized_manufacturer", "breast_density_score", "view"], inplace=True
    )
    train_data = train_data.merge(image_meta, on="patient_id", how="inner")
    logger.info(f"{len(train_data) = :_}")

    # Create folds
    gkf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=14365)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X=train_data, y=train_data)):
        folds.append((train_idx, val_idx))

    # Save train and test sets to the output directory
    train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    # Training
    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        logger.info(f"{fold_idx = :_}")
        train_image_uids = set(image_meta["image_uid"][train_idx].values)
        valid_image_uids = set(image_meta["image_uid"][val_idx].values)
        logger.info(f"{len(train_image_uids) = :_}")
        logger.info(f"{len(valid_image_uids) = :_}")

        train_dataset = Dataset(
            image_meta=image_meta,
            annotations_df=annotations_df,
            image_uids=train_image_uids,
            random_aug=True,
            cache=True,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)

        valid_dataset = Dataset(
            image_meta=image_meta,
            annotations_df=annotations_df,
            image_uids=valid_image_uids,
            random_aug=False,
            cache=True,
        )

        valid_dataloader = DataLoader(valid_dataset, batch_size=16, num_workers=4)

        model = MammoDlModule()

        checkpoints = [
            ModelCheckpoint(
                filename=f"{{VALID_loss}}-fold{fold_idx}-{{epoch}}",
                monitor="VALID_loss",
                save_top_k=1,
                mode="min",
            ),
            ModelCheckpoint(
                filename=f"{{VALID_dense_loss}}-fold{fold_idx}-{{epoch}}",
                monitor="VALID_dense_loss",
                save_top_k=1,
                mode="min",
            ),
            ModelCheckpoint(
                filename=f"{{VALID_breast_loss}}-fold{fold_idx}-{{epoch}}",
                monitor="VALID_breast_loss",
                save_top_k=1,
                mode="min",
            ),
            ModelCheckpoint(filename=f"final-fold{fold_idx}-{{epoch}}", save_last=True),
            EarlyStopping(monitor="VALID_dense_loss", patience=15, mode="min"),
        ]

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=10,
            check_val_every_n_epoch=1,
            log_every_n_steps=23,
            default_root_dir=os.path.join(output_dir, f"fold-{fold_idx}"),
            callbacks=checkpoints,
        )

        trainer.fit(model=model, train_dataloaders=[train_dataloader], val_dataloaders=[valid_dataloader])


def avg_model_checkpoints(output_dir: Path) -> None:
    list_valid_loss = []
    list_fold_value = []
    checkpoint_paths = []
    for fold_idx in range(1, K_FOLDS):
        content = glob.glob(
            os.path.join(output_dir, f"fold-{fold_idx}", "lightning_logs/version_0/checkpoints", "*.ckpt")
        )
        search_element = "VALID_dense_loss"
        path = []
        for string in content:
            if search_element in string:
                path = string
        checkpoint_paths.append(path)

        name_val_loss = path.split("/")[-1]
        name_val_loss = name_val_loss.split(".ckpt")[0]
        pattern_1 = r"=(.*?)-"
        pattern_2 = r"-fold(.*?)-"
        valid_loss_value = round(float(re.findall(pattern_1, name_val_loss)[0]), 6)
        fold_value = int(re.findall(pattern_2, name_val_loss)[0])
        list_valid_loss.append(valid_loss_value)
        list_fold_value.append(fold_value)

    valid_loss_dict = {"valid_loss_value": list_valid_loss, "fold_value": list_fold_value}
    data = pd.DataFrame(valid_loss_dict)
    best_valid_loss = min(data.valid_loss_value)
    fold_min_loss = int(data.fold_value[data.valid_loss_value == best_valid_loss].tolist()[0])
    logger.info(f"{best_valid_loss = :_}")
    logger.info(f"{fold_min_loss = :_}")

    os.mkdir(os.path.join(output_dir, "best_fold"))
    source_directory = os.path.join(
        output_dir, f"fold-{fold_min_loss}", "lightning_logs/version_0/checkpoints"
    )
    destination_directory = os.path.join(output_dir, "best_fold")
    files = os.listdir(source_directory)
    for file in files:
        source_path = os.path.join(source_directory, file)
        destination_path = os.path.join(destination_directory, file)
        shutil.copy(source_path, destination_path)

    kfold_checkpoints = []
    for checkpoint_path in checkpoint_paths:
        model = MammoDlModule.load_from_checkpoint(checkpoint_path)
        kfold_checkpoints.append(model.state_dict())

    averaged_checkpoints = {}
    for key in kfold_checkpoints[0].keys():
        averaged_checkpoints[key] = torch.mean(
            torch.stack([weight[key].float() for weight in kfold_checkpoints]), dim=0
        )

    os.mkdir(os.path.join(output_dir, "kfold_averaged_checkpoints"))
    torch.save(
        averaged_checkpoints,
        os.path.join(output_dir, "kfold_averaged_checkpoints/averaged_checkpoints_valid_dense_loss.ckpt"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("-h5", "--h5_dir", dest="h5_dir", type=Path, required=True)
    parser.add_argument("-anno", "--annotations", dest="annotations", type=Path, required=True)
    params = parser.parse_args(sys.argv[1:])

    if not params.output_dir.exists():
        params.output_dir.mkdir(parents=True)

    set_logger()

    train(output_dir=params.output_dir, h5_dir=params.h5_dir, annotations=params.annotations)

    avg_model_checkpoints(output_dir=params.output_dir)
