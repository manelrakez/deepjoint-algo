# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

output_dir="~/output_dir/Dataset_in_H5_dir" # Specify the output path for the H5 files
dicom_dir="~/Dataset_in_DICOM_dir"  # Specify the path to input images stored in DICOM format

export PYTHONPATH="${DIRECTORY}/src"

python "~/deepjoint-algo/scripts/deepjoint_torch/dcm_to_h5.py" \
--dicom_dir "${dicom_dir}" \
--output_dir "${output_dir}"

