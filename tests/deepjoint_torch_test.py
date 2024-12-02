# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

from pathlib import Path

from deepjoint_torch.constants import BEST_CHECKPOINT

from deepjoint_torch.infer import infer
from deepjoint_torch.h5 import load_image_meta
from deepjoint_torch.dataset import Dataset
from deepjoint_torch import TEST_DATA_DIR



def test_dataset():
    """Iterate over the dataset and assert that image and masks have the same shape"""

    image_meta = load_image_meta(h5_dir=TEST_DATA_DIR / "deepjoint_torch" / "h5_dir")

    dataset = Dataset(
        image_meta=image_meta,
        annotations_df=None,
        random_aug=False,
        cache=False,
    )

    for items in dataset:
        image_uid, image, breast_mask, dense_mask = items
        assert image.shape[0] == breast_mask.shape[0] == dense_mask.shape[0], f"Issue with {image_uid = }"
        assert image.shape[1] == breast_mask.shape[1] == dense_mask.shape[1], f"Issue with {image_uid = }"


def test_infer(tmp_path: Path) -> None:
    """Test that the inference pipeline is working on test data"""
    infer(
        output_dir=tmp_path,
        h5_dir=TEST_DATA_DIR / "deepjoint_torch" / "h5_dir",
        checkpoint=BEST_CHECKPOINT,
        eval_model=False
    )


def test_infer_with_eval(tmp_path: Path) -> None:
    """Test that the inference pipeline is working on test data"""
    infer(
        output_dir=tmp_path,
        h5_dir=TEST_DATA_DIR / "deepjoint_torch" / "h5_dir",
        checkpoint=BEST_CHECKPOINT,
        annotations=TEST_DATA_DIR / "deepjoint_torch" / "annotations.csv",
        eval_model=True
    )