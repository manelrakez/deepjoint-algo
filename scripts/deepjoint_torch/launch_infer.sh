# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

DIRECTORY="~/deepjoint-algo" # Specify the path to the "deepjoint-algo" directory
output_dir="~/output_dir/inference" # Specify the output path for the inference step
h5_dir="~/Dataset_in_h5_dir"  # Specify the path to input images stored in .h5 format

export PYTHONPATH="${DIRECTORY}/src"
export CUDA_VISIBLE_DEVICES="6" #Specify your GPU

python "~/deepjoint-algo/scripts/deepjoint_torch/infer.py" \
--output_dir "${output_dir}" \
--h5_dir "${h5_dir}"

#If inference step: Compute visit-level metrics to use in the joint model part
python "~/deepjoint-algo/scripts/deepjoint_torch/visit-level_metrics_computation.py" \
--output_dir "${output_dir}" \
--h5_dir "${h5_dir}"