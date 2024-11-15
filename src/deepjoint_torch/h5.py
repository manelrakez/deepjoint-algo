# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

"""The reader module defines methods and classes that can be used to read H5 file
data in a friendly and predictable way.
"""

from pathlib import Path
import h5py


from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataSample:
    image: np.ndarray
    breast_mask: np.ndarray
    crop_pad_direction: str
    photometric_interpretation: str
    window_level: int
    window_width: int
    voi_lut_function: str
    breast_min: int
    breast_max: int
    breast_mean: float
    breast_std: float
    image_min: int
    image_max: int
    annotations: dict[str, np.ndarray] | None = None

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


def read_h5(h5_path: Path) -> DataSample:
    with h5py.File(h5_path.as_posix(), mode="r", swmr=True) as opened_h5:
        image = opened_h5.get("image")
        breast_mask = opened_h5.get("mask")

        # read_buffer:
        image = image[:]
        breast_mask = breast_mask[:]

        crop_pad_direction = opened_h5.get("crop_pad_direction")[()].decode("utf-8")
        breast_min = opened_h5.get("breast_min")[()]
        breast_max = opened_h5.get("breast_max")[()]
        breast_mean = opened_h5.get("breast_mean")[()]
        breast_std = opened_h5.get("breast_std")[()]
        image_min = opened_h5.get("image_min")[()]
        image_max = opened_h5.get("image_max")[()]

        window_level, window_width = opened_h5.get("windowing")[:].ravel()

        voi_lut_function = opened_h5.get("voi_lut_function")[()].decode("utf-8")
        photometric_interpretation = opened_h5.get("photometric_interpretation")[()].decode("utf-8")

    return DataSample(
        image=image,
        breast_mask=breast_mask,
        crop_pad_direction=crop_pad_direction,
        photometric_interpretation=photometric_interpretation,
        window_level=window_level,
        window_width=window_width,
        voi_lut_function=voi_lut_function,
        breast_min=breast_min,
        breast_max=breast_max,
        breast_mean=breast_mean,
        breast_std=breast_std,
        image_min=image_min,
        image_max=image_max,
    )


def load_image_meta(h5_dir: Path) -> pd.DataFrame:
    image_meta = pd.read_csv(h5_dir / "tags.csv")
    image_meta.set_index("image_uid", inplace=True, drop=False)
    image_meta.index.name = ""

    def _add_h5_file(_image_uid: str) -> Path:
        _h5_file = h5_dir / f"{_image_uid}.h5"
        if not _h5_file.exists():
            raise FileNotFoundError(f"{_h5_file = } for {_image_uid = }")
        return _h5_file

    # Add 'h5_path' column
    image_meta["h5_path"] = image_meta["image_uid"].apply(_add_h5_file)

    image_meta["pixel_spacing"] = image_meta["pixel_spacing"].apply(
        lambda raw_ps: tuple(float(val) for val in raw_ps.split(","))
    )
    return image_meta
