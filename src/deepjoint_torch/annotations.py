# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

import json
import os

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from loguru import logger


BREAST_LABEL = 0
DENSE_LABEL = 4


def load_annotations(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df["image_uid"] = df["fp"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df["polygons"] = df["polygons"].str.replace("label", "type").str.replace("polygon", "point_list")
    df.rename(columns={"fp": "hdf5_path"}, inplace=True)

    # Don't keep empty annotations
    image_uids_without_annot = df[df["polygons"] == "[]"]["image_uid"].unique().tolist()
    if len(image_uids_without_annot) > 1:
        logger.warning(
            f"{len(image_uids_without_annot):_} images don't have annotations and will not be processed"
        )
        df = df[~df["image_uid"].isin(image_uids_without_annot)]

    logger.success(f"A total of {len(df):_} images have annotations")
    columns_to_keep = ["image_uid", "hdf5_path", "polygons"]

    return df[columns_to_keep]


def get_masks_from_polygons(
    polygons: List[Dict[str, Any]] | str,
    im_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (H, W, 5) numpy ndarray.

    # 0 -> breast      -> breast_mask
    # 1 -> pectoral    -> not used
    # 2 -> mass        -> not used
    # 3 -> background  -> not used
    # 4 -> Dense_T     -> dense_mask
    """
    valid_polygon_types = [BREAST_LABEL, DENSE_LABEL]

    if isinstance(polygons, str):
        polygons = json.loads(polygons)

    vertices_dict = {polygon_type: [] for polygon_type in valid_polygon_types}

    for polygon in polygons:
        polygon_type = int(polygon["type"])
        if polygon_type not in valid_polygon_types:
            logger.warning(f"Found a polygons with {polygon_type = }. Not in {valid_polygon_types = }")
            continue
        vertices_dict[polygon_type].append(np.array(polygon["point_list"], dtype=np.int32))

    breast_mask = np.zeros(im_shape, dtype=np.uint8)
    if len(vertices_dict[BREAST_LABEL]) > 0:
        vertices = vertices_dict[BREAST_LABEL]
        # logger.debug(f"Add {len(vertices) = } to breast_mask with shape {breast_mask.shape}")
        cv2.fillPoly(breast_mask, pts=vertices, color=1)  # noqa

    dense_mask = np.zeros(im_shape, dtype=np.uint8)
    if len(vertices_dict[DENSE_LABEL]) > 0:
        vertices = vertices_dict[DENSE_LABEL]
        # logger.debug(f"Add {len(vertices) = } to dense_mask with shape {dense_mask.shape}")
        cv2.fillPoly(dense_mask, pts=vertices, color=1)  # noqa

    return breast_mask, dense_mask
