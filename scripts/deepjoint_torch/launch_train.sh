# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

DIRECTORY="~/deepjoint-algo" # Specify the path to the "deepjoint-algo" directory
output_dir="~/output_dir/train" # Specify the output path for the training step
h5_dir="~/Dataset_in_h5_dir"  # Specify the path to input images stored in .h5 format
annotations="~/annotations/annon.csv" # Specify the complete path to image annotations.csv

export PYTHONPATH="${DIRECTORY}/src"

CUDA_VISIBLE_DEVICES="0" python "~/deepjoint-algo/scripts/deepjoint_torch/train.py" \
--output_dir "${output_dir}"\
--h5_dir "${h5_dir}"\
--annotations "${annotations}"