# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.


import json
import re

from datetime import datetime

from deepjoint_torch.h5 import  read_h5
from deepjoint_torch.transforms import annotation_transforms
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px

from dash import Dash, Input, Output, State, callback, callback_context
from loguru import logger

from annot_tool.components import (
    get_annotation_component,
    get_data_store,
    get_image_component,
    get_metadata_component,
)
from annot_tool.constants import DEFAULT_DATA_VALUES
from annot_tool.data import save_data_to_csv


def get_app(image_meta: pd.DataFrame, start_data: dict, output_file: str) -> Dash:
    external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/image_annotation_style.css"]
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    metadata_card = get_metadata_component(image_meta)
    annotation_card = get_annotation_component()
    image_graph = get_image_component()
    data_store = get_data_store(start_data=start_data)

    # layout = html.Div([metadata_card])
    layout = html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col([metadata_card, image_graph], md=7),
                            dbc.Col([annotation_card], md=5),
                            data_store,
                        ],
                    ),
                ],
                fluid=True,
            ),
        ]
    )

    @callback(
        Output("metadata_table", "selected_rows"),
        [
            Input("previous", "n_clicks"),
            Input("next", "n_clicks"),
        ],
        [
            State("metadata_table", "derived_virtual_indices"),
            State("metadata_table", "derived_virtual_selected_rows"),
            State("metadata_table", "selected_rows"),
        ],
    )
    def previous_next_selected_row(
        previous_n_clicks,
        next_n_clicks,
        derived_virtual_indices,
        derived_virtual_selected_rows,
        selected_rows,
    ):
        logger.debug("callback previous_next() : ")
        logger.debug(f"previous_n_clicks : {previous_n_clicks} | {type(previous_n_clicks)}")
        logger.debug(f"next_n_clicks : {next_n_clicks} | {type(next_n_clicks)}")

        logger.debug(f"derived_virtual_indices : {derived_virtual_indices} | {type(derived_virtual_indices)}")
        logger.debug(
            f"derived_virtual_selected_rows : {derived_virtual_selected_rows} | {type(derived_virtual_selected_rows)}"
        )

        logger.debug(f"selected_rows : {selected_rows} | {type(selected_rows)}")

        cbcontext = [p["prop_id"] for p in callback_context.triggered][0]
        logger.debug(f"cbcontext : {cbcontext} | {type(cbcontext)}")

        if cbcontext not in ["previous.n_clicks", "next.n_clicks"]:
            return []

        current_selected_row = selected_rows[0]
        logger.debug(f"current_selected_row : {current_selected_row} | {type(current_selected_row)}")
        current_derived_virtual_selected_row_index = np.argwhere(
            np.asarray(derived_virtual_indices) == current_selected_row
        )[0][0]
        logger.debug(
            f"current_derived_virtual_selected_row_index : {current_derived_virtual_selected_row_index} | {type(current_derived_virtual_selected_row_index)}"
        )

        new_derived_virtual_selected_row_index = current_derived_virtual_selected_row_index

        # new_active_cell = active_cell.copy()
        if cbcontext == "previous.n_clicks":
            # new_active_cell["row"] = (active_cell["row"] - 1) % metadata.shape[0]
            new_derived_virtual_selected_row_index = (current_derived_virtual_selected_row_index - 1) % len(
                derived_virtual_indices
            )
        if cbcontext == "next.n_clicks":
            # new_active_cell["row"] = (active_cell["row"] + 1) % metadata.shape[0]
            new_derived_virtual_selected_row_index = (current_derived_virtual_selected_row_index + 1) % len(
                derived_virtual_indices
            )
        # new_active_cell["row_id"] = derived_virtual_row_ids[new_active_cell["row"]]
        # return (new_active_cell,)
        logger.debug(
            f"new_derived_virtual_selected_row_index :{new_derived_virtual_selected_row_index} | {type(new_derived_virtual_selected_row_index)}"
        )
        new_selected_row = derived_virtual_indices[new_derived_virtual_selected_row_index]
        logger.debug(f"new_selected_row : {new_selected_row} | {type(new_selected_row)}")

        return [
            new_selected_row,
        ]

    @callback(
        Output("image_graph", "figure"),
        Input("metadata_table", "derived_virtual_selected_row_ids"),
        State("all-data", "data"),
    )
    def update_image_figure(derived_virtual_selected_row_ids, all_data):
        logger.debug("callback update_image_figure() : ")
        logger.debug(f"derived_virtual_selected_row_ids : {derived_virtual_selected_row_ids}")

        if derived_virtual_selected_row_ids is None or len(derived_virtual_selected_row_ids) == 0:
            img = np.zeros((512, 512), dtype=np.uint8)
            shapes = []

        else:
            image_uid = derived_virtual_selected_row_ids[0]
            try:
                shapes = all_data[image_uid]["shapes"]
                h5_path = image_meta.at[image_uid, "h5_path"]
                img = annotation_transforms(read_h5(h5_path)).image

                # img = np.squeeze(sample.image, axis=-1)  # remove last axis (H,W,1) -> (H,W)
            except (Exception,) as e:
                logger.warning(f"Impossible to load image for ImageUID={image_uid}")
                logger.debug(str(e))
                img = np.zeros((512, 512), dtype=np.uint8)
                shapes = []

        new_fig = px.imshow(
            img,
            binary_backend="jpg",
            color_continuous_scale="gray",
            height=img.shape[0] // 2,
            width=img.shape[1] // 2,
        )
        new_fig.update_layout(
            shapes=shapes,
            # reduce space between image and graph edges
            margin=dict(l=0, r=0, b=0, t=0, pad=4),
            dragmode="drawclosedpath",
        )
        return new_fig

    @callback(
        Output("all-data", "data"),
        [Input("image_graph", "relayoutData")],
        [
            State("metadata_table", "derived_virtual_selected_row_ids"),
            State("all-data", "data"),
            State("radio-annot-type", "value"),
        ],
    )
    def update_data_store(
        graph_relayout_data,
        derived_virtual_selected_row_ids,
        all_data,
        annot_type_value,
    ):
        logger.debug("callback update_data_store() : ")
        logger.debug(f"graph_relayout_data : {graph_relayout_data} | {type(graph_relayout_data)}")
        logger.debug(
            f"derived_virtual_selected_row_ids : {derived_virtual_selected_row_ids} | {type(derived_virtual_selected_row_ids)}"
        )
        logger.debug(f"annot_type_value : {annot_type_value} | {type(annot_type_value)}")
        # logger.debug(f"all_data : {all_data} | {type(all_data)}")

        cbcontext = [p["prop_id"] for p in callback_context.triggered][0]
        logger.debug(f"cbcontext : {cbcontext} | {type(cbcontext)}")

        if derived_virtual_selected_row_ids is None or len(derived_virtual_selected_row_ids) == 0:
            logger.debug("No selected row")
            return dash.no_update

        image_id = derived_virtual_selected_row_ids[0]

        if cbcontext == "image_graph.relayoutData":
            logger.debug("Try to add new annotation ... ")
            keys = list(graph_relayout_data.keys())
            logger.debug(f"{keys = }")

            # for k, v in graph_relayout_data.items():
            #   print(type(k), k)
            #    print(type(v), v)

            if "shapes" in keys:
                # New annotation
                logger.debug(f"Add new annotation : {graph_relayout_data}")
                shapes = graph_relayout_data["shapes"]
                all_data[image_id]["shapes"] = shapes
                all_data[image_id]["annot_type_labels"].append(annot_type_value)

            elif re.match("shapes\[[0-9]+\].path", keys[0]):
                # Update existing annotation
                key = keys[0]
                ind = int(key.split("[")[-1].split("]")[0])
                logger.debug("Update shape : ", ind)
                path = graph_relayout_data[key]
                shapes = all_data[image_id]["shapes"]
                shapes[ind]["path"] = path
                all_data[image_id]["shapes"] = shapes

            return all_data
        else:
            return dash.no_update

    @app.callback(
        Output("display-internal-data", "children"),
        [
            Input("metadata_table", "derived_virtual_selected_row_ids"),
            Input("all-data", "data"),
        ],
    )
    def update_internal_data(derived_virtual_selected_row_ids, all_data):
        logger.debug("callback update_internal_data() : ")
        logger.debug(f"derived_virtual_selected_row_ids : {derived_virtual_selected_row_ids}")
        # logger.debug(f"all_data : {all_data}")

        if derived_virtual_selected_row_ids is None or len(derived_virtual_selected_row_ids) == 0:
            return json.dumps(DEFAULT_DATA_VALUES, indent=2)

        image_uid = derived_virtual_selected_row_ids[0]
        to_display = all_data[image_uid]
        to_display["paths"] = [shape["path"] for shape in to_display["shapes"]]  # display only 'path'
        to_display.pop("shapes")

        return json.dumps(to_display, indent=2)

    @app.callback(
        Output("display-logs", "children"),
        [
            Input("metadata_table", "derived_virtual_selected_row_ids"),
            Input("save", "n_clicks"),
        ],
        State("all-data", "data"),
    )
    def save_and_update_logs(derived_virtual_selected_row_ids, save_button_clicks, all_data):
        logger.debug("save_and_update_logs()")
        logger.debug(f"derived_virtual_selected_row_ids : {derived_virtual_selected_row_ids}")
        logger.debug(f"save_button_clicks : {save_button_clicks}")
        # logger.debug(f"all_data : {all_data}")

        cbcontext = [p["prop_id"] for p in callback_context.triggered][0]

        save_data_to_csv(all_data, output_file=output_file)

        if derived_virtual_selected_row_ids is None or len(derived_virtual_selected_row_ids) == 0:
            return "No selected row"

        if cbcontext == "save.n_clicks":
            return f"{datetime.now()} manual save. Selected ImageID={derived_virtual_selected_row_ids[0]}"
        else:
            return f"{datetime.now()} automatic save. Selected ImageID={derived_virtual_selected_row_ids[0]}"

    app.layout = layout
    return app
