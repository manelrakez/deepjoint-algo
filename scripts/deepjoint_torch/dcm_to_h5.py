import sys
from pathlib import Path
from argparse import ArgumentParser
from deepjoint_torch.dcm import dcm_to_h5

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dicom_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    params = parser.parse_args(sys.argv[1:])
    dcm_to_h5(dicom_dir=params.dicom_dir, output_dir=params.output_dir)
