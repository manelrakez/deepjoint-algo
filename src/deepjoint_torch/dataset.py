# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

from copy import deepcopy
from typing import Tuple, Any

import numpy as np
import pandas as pd
import torch

from loguru import logger
from torch.utils import data

from deepjoint_torch.annotations import BREAST_LABEL, DENSE_LABEL, get_masks_from_polygons

from deepjoint_torch.h5 import DataSample, read_h5
from deepjoint_torch.transforms import transforms


class Dataset(data.Dataset):
    def __init__(
        self,
        image_meta: pd.DataFrame,
        annotations_df: pd.DataFrame | None = None,
        image_uids: set[str] | None = None,
        random_aug: bool = False,
        cache: bool = False,
    ):
        # data-selection:
        # * 'image_uid' : should be in 'image_meta'
        # * CC and MLO must have a non-empty 'breast_mask'
        #   and may have an empty 'dense_mask' (extremely low dense breast)

        if image_uids is None:
            image_uids = set(image_meta["image_uid"].values)

        if annotations_df is not None:
            image_uids_not_in_annotations_df = image_uids.difference(annotations_df["image_uid"].values)
            if len(image_uids_not_in_annotations_df):
                logger.warning(
                    f"{len(image_uids_not_in_annotations_df) = : _} image_uid(s) not in annotations_df"
                )
                image_uids = image_uids.intersection(annotations_df["image_uid"].values)

            # do a .copy() to do not edit original 'image_meta'. Keep only rows of interest
            annotations_df = annotations_df[annotations_df["image_uid"].isin(image_uids)].copy()

            annotations_df["with_breast_mask"], annotations_df["with_dense_mask"] = zip(
                *annotations_df["polygons"].map(infer_with_masks)
            )
            logger.debug(
                f"counts for 'with_breast_mask' : {annotations_df['with_breast_mask'].value_counts().to_dict()}"
            )
            logger.debug(
                f"counts for 'with_dense_mask' : {annotations_df['with_dense_mask'].value_counts().to_dict()}"
            )

            if False in annotations_df["with_breast_mask"].values:
                image_uids_without_breast_mask = set(
                    annotations_df[~annotations_df["with_breast_mask"]]["image_uid"].values
                )
                logger.warning(
                    f"{len(image_uids_without_breast_mask) = :_} image_uid(s) without 'breast_mask' in annots"
                )
                image_uids = image_uids.difference(image_uids_without_breast_mask)

        # set dataset attributes now:
        self.image_uids = sorted(list(image_uids))
        self.image_meta = image_meta[image_meta["image_uid"].isin(image_uids)].copy()

        if annotations_df is not None:
            self.annotations_df = annotations_df[annotations_df["image_uid"].isin(image_uids)]
            assert self.annotations_df["image_uid"].is_unique
            self.annotations_df.set_index("image_uid", inplace=True, drop=False)
            self.annotations_df.index.name = ""
        else:
            self.annotations_df = None

        self.random_aug = random_aug
        self.cache = cache
        self.__cache = {} if cache else None

    def __len__(self):
        return len(self.image_uids)

    def __getitem__(self, index: int):
        image_uid = self.image_uids[index]
        h5_file = self.image_meta.at[image_uid, "h5_path"]
        if self.cache and image_uid in self.__cache:
            # get 'sample' already 'transformed' with breast & dense masks transformed too
            sample: DataSample = deepcopy(self.__cache[image_uid])
        else:
            sample = read_h5(h5_file)
            im_shape = sample.height, sample.width
            # get 'breast_mask' and 'dense_mask'

            if self.annotations_df is not None:
                breast_mask, dense_mask = get_masks_from_polygons(
                    polygons=self.annotations_df.at[image_uid, "polygons"],
                    im_shape=im_shape,
                )
            else:
                breast_mask = np.zeros(im_shape, dtype=np.uint8)
                dense_mask = np.zeros(im_shape, dtype=np.uint8)

            sample.annotations = {"breast_mask": breast_mask, "dense_mask": dense_mask}
            sample = transforms(sample)

            if self.cache:
                # add to cache transformed data sample
                self.__cache[image_uid] = deepcopy(sample)

        if self.random_aug:
            # TODO : add here random augmentation
            pass

        image = sample.image
        breast_mask = sample.annotations["breast_mask"]
        dense_mask = sample.annotations["dense_mask"]

        # need to create image & true masks  with shape (1, H, W) : add new axis
        # making images with shape (3, H, W) is done in the model now.
        image = np.expand_dims(image, axis=0)
        breast_mask = np.expand_dims(breast_mask, axis=0)
        dense_mask = np.expand_dims(dense_mask, axis=0)

        # cast to torch.Tensor with good dtype
        image = torch.from_numpy(image).type(torch.float32)
        breast_mask = torch.from_numpy(breast_mask).type(torch.long)
        dense_mask = torch.from_numpy(dense_mask).type(torch.long)
        return image_uid, image, breast_mask, dense_mask


def infer_with_masks(polygons: dict[str, Any]) -> Tuple[bool, bool]:
    # polygons = json.loads(polygons)
    with_breast_mask, with_dense_mask = False, False

    for polygon in polygons:
        polygon_type = int(polygon["type"])
        if polygon_type == BREAST_LABEL:
            with_breast_mask = True
        if polygon_type == DENSE_LABEL:
            with_dense_mask = True

    return with_breast_mask, with_dense_mask
