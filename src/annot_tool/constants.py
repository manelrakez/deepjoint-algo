# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.


HOST = "0.0.0.0"
PORT = 5001
DEBUG = True


DEFAULT_SHAPE_DICT = {
    "editable": True,
    "xref": "x",
    "yref": "y",
    "layer": "above",
    "opacity": 1,
    "line": {"color": "#444", "width": 4, "dash": "solid"},
    "fillcolor": "rgba(0,0,0,0)",
    "fillrule": "evenodd",
    "type": "path",
    "path": "",
}

DEFAULT_DATA_VALUES = {
    "shapes": [],  # list[dict[str, Any]]
    "annot_type_labels": [],  # list[int]
}


ANNOT_TYPES = ["breast_mask", "pectoral", "mass", "background", "dense_mask"]
ANNOT_TYPES_MAPPING = {annot_type: i for i, annot_type in enumerate(ANNOT_TYPES)}
ANNOT_TYPES_REVERSE_MAPPING = {i: annot_type for i, annot_type in enumerate(ANNOT_TYPES)}