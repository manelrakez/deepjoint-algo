# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

DIRECTORY="~/deepjoint-algo" # Specify the path to the "deepjoint-algo" directory
output_dir="~/output_dir/eval" # Specify the output path for the inference step
h5_dir="~/Dataset_in_H5_dir"  # Specify the path to input images stored in .h5 format
annotations="~/annotations/annon.csv" # Specify the complete path to image annotations.csv

export PYTHONPATH="${DIRECTORY}/src"
export CUDA_VISIBLE_DEVICES="6" #Specify your GPU

python "~/deepjoint-algo/scripts/deepjoint_torch/infer.py" \
--output_dir "${output_dir}" \
--h5_dir "${h5_dir}" \
--annotations "${annotations}" \
--eval_mode
