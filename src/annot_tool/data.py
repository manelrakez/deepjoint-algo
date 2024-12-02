# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.


import json
from copy import deepcopy

import  pandas as pd

from deepjoint_torch.annotations import load_annotations
from annot_tool.constants import DEFAULT_DATA_VALUES, DEFAULT_SHAPE_DICT



def save_data_to_csv(all_data:dict, output_file:str)->None:
    to_save = []
    for image_uid, data in all_data.items():

        polygons = []
        for shape, annot_type_label in zip(data["shapes"], data["annot_type_labels"]):
            path:str =  shape["path"]
            point_list = path_to_point_list(path)
            polygons.append({"type": annot_type_label, "point_list": point_list})

        # to_save.append({"image_uid": image_uid, "polygons": polygons})
        to_save.append({"image_uid": image_uid, "polygons": json.dumps(polygons)})

    to_save= pd.DataFrame(to_save)
    to_save.to_csv(output_file, index=False, header=True)


def load_existing_annotations(output_file:str) -> dict:
    annotations_df = load_annotations(output_file, remove_images_without_annots=False)
    start_data = dict()

    for image_uid, row in annotations_df.iterrows():

        _data = deepcopy(DEFAULT_DATA_VALUES)

        if len(row["polygons"]) != 0:
            annot_type_labels, shapes = [], []
            for polygon in row["polygons"]:

                annot_type_labels.append(polygon["type"])

                point_list = polygon["point_list"]
                path = point_list_to_path(point_list)
                shape = deepcopy(DEFAULT_SHAPE_DICT)
                shape["path"] = path
                shapes.append(shape)

            _data["shapes"] = shapes
            _data["annot_type_labels"] =annot_type_labels

        start_data[image_uid] = _data

    return start_data


def path_to_point_list(path:str) -> list[list[float]]:
    path = path.replace("M", "").replace("Z", "")

    point_list = []
    for xy in path.split('L'):
        x, y = xy.split(",")
        x, y = float(x), float(y)
        point_list.append([x, y])

    return point_list

def point_list_to_path(point_list:list[list[float]]) -> str:

    all_xy = []
    for (x,y) in point_list:
        all_xy.append(f"{x},{y}")

    path = "M"+"L".join(all_xy)+"Z"
    return path