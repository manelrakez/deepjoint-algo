# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.


import json

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px

from dash import dash_table
from annot_tool.constants import ANNOT_TYPES_MAPPING

DEFAULT_TO_DISPLAY = {"current_image_uid": "None"}


def get_metadata_component(metadata: pd.DataFrame) -> dbc.Card:
    metadata["id"] = metadata["image_uid"]
    metadata_columns = ["id", "image_uid"]
    # fmt: off
    table_columns = [
        {"name": "ImageUID", "id": "image_uid", "type": "text"},
    ]
    # fmt: on

    metadata_table = dash_table.DataTable(
        data=metadata[metadata_columns].to_dict("records"),
        sort_action="native",
        filter_action="native",
        columns=table_columns,
        page_action="none",  # or page_size=5
        style_table={"height": "200px", "overflowY": "auto"},
        style_cell={"minWidth": 25, "maxWidth": 150, "width": "auto"},
        style_header={"fontWeight": "bold"},
        fixed_rows={"headers": True},
        row_selectable="single",
        selected_rows=[
            0,
        ],
        persisted_props=["selected_rows", "filter_query", "sort_by"],
        id="metadata_table",
    )

    buttons = dbc.ButtonGroup(
        [
            dbc.Button("Previous ImageUID", id="previous", outline=True),
            dbc.Button("Next ImageUID", id="next", outline=True),
        ],
        size="lg",
        style={"width": "100%"},
    )
    body = dbc.CardBody([metadata_table])
    footer = dbc.CardFooter([buttons])

    return dbc.Card(id="metadata_card", children=[body, footer])


def get_image_component() -> dcc.Graph:
    image_fig = px.imshow(
        np.zeros((512, 512), dtype=np.uint8), binary_backend="jpg", color_continuous_scale="gray"
    )
    image_fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        dragmode="drawclosedpath",
        height=300,
        width=200,
    )

    image_graph = dcc.Graph(
        id="image_graph",
        figure=image_fig,
        config={
            "modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"],
            "modeBarButtonsToRemove": [
                "resetScale2d",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
                "toggleSpikelines",
            ],
            "editable": False,
        },
    )

    return image_graph


def get_annotation_component() -> dbc.Card:
    annotation_card = dbc.Card(
        [
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(html.H2("Metadata"), md=7),
                        dbc.Col(
                            dbc.Button(
                                "Save annotations",
                                id="save",
                                outline=True,
                            ),
                            md=4,
                        ),
                    ]
                )
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        dbc.Col(
                            [
                                dcc.RadioItems(
                                    id="radio-annot-type",
                                    options=[
                                        {"label": annot_type, "value": value}
                                        for (annot_type, value) in ANNOT_TYPES_MAPPING.items()
                                    ],
                                    labelStyle={
                                        "display": "inline-block",
                                        "padding": "0px 12px 12px 0px",
                                    },
                                    value=ANNOT_TYPES_MAPPING["breast_mask"],  # breast_mask as default
                                ),
                                html.H5("Internal data "),
                                html.Pre(
                                    id="display-internal-data",
                                    children=json.dumps(DEFAULT_TO_DISPLAY, indent=2),
                                ),
                                html.H5("Logs "),
                                html.Pre(id="display-logs", children=""),
                            ],
                        ),
                    )
                ]
            ),
        ],
    )
    return annotation_card


def get_data_store(start_data: dict) -> dcc.Store:
    return dcc.Store(id="all-data", data=start_data)
