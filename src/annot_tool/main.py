# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.


import argparse
import os
import random
import string
from pathlib import Path
from _datetime import datetime
from loguru import logger

from annot_tool.app import get_app
from annot_tool.constants import DEBUG, HOST, PORT, DEFAULT_DATA_VALUES
from annot_tool.data import load_existing_annotations
from deepjoint_torch.h5 import load_image_meta


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--h5_dir", dest="h5_dir", required=True, type=Path)
    argument_parser.add_argument(
        "-f",
        "--file",
        dest="output_file",
        default="annotations_{}_{}.csv".format(
            datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
            "".join(random.choices(string.ascii_uppercase + string.digits, k=6)),
        ),
        type=str,
    )

    params = argument_parser.parse_args()
    logger.debug(f"{params = }")

    image_meta = load_image_meta(params.h5_dir)
    if os.path.isfile(params.output_file):
        start_data = load_existing_annotations(params.output_file) # pd.DataFrame with columns
    else:
        start_data = {image_uid: DEFAULT_DATA_VALUES for image_uid in image_meta["image_uid"].values}

    app = get_app(image_meta=image_meta, start_data=start_data, output_file=params.output_file)
    app.server.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == "__main__":
    main()
