from pathlib import Path

import pandas as pd
from deepjoint_torch.constants import BEST_CHECKPOINT
from deepjoint_torch.cli.dcm_to_h5 import dcm_to_h5
from deepjoint_torch.h5 import read_h5
from deepjoint_torch import TEST_DATA_DIR
from tqdm.auto import tqdm
from deepjoint_torch.transforms import transforms
import pytest
from deepjoint_torch.constants import IMAGE_WIDTH, IMAGE_HEIGHT
from deepjoint_torch.h5 import load_image_meta
from deepjoint_torch.dataset import Dataset
from deepjoint_torch.cli.infer import infer


@pytest.fixture(name="h5_dir", scope="module")
def test_dicom_to_h5(tmp_path_factory) -> Path:
    h5_dir = tmp_path_factory.mktemp("h5_dir")
    dcm_to_h5(dicom_dir=TEST_DATA_DIR, output_dir=h5_dir)
    return h5_dir


@pytest.fixture(name="image_meta", scope="module")
def test_load_image_meta(h5_dir: Path) -> pd.DataFrame:
    return load_image_meta(h5_dir=h5_dir)


def test_read_h5_and_transform(h5_dir: Path):
    for h5_file in tqdm(h5_dir.glob("**/*.h5"), desc="Test read_hdf5()"):
        data_sample = read_h5(h5_file)
        data_sample = transforms(data_sample)
        assert data_sample.width == IMAGE_WIDTH
        assert data_sample.height == IMAGE_HEIGHT


def test_dataset(image_meta: pd.DataFrame):
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


def test_infer(h5_dir: Path, tmp_path: Path) -> None:
    infer(output_dir=tmp_path, h5_dir=h5_dir, checkpoint=BEST_CHECKPOINT, eval_model=False)
