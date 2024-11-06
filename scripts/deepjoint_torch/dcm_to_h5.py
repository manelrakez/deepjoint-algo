import sys
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

from deepjoint_torch.dcm import DCM
from tqdm.auto import tqdm
from shutil import rmtree
from loguru import logger


def dcm_to_h5(dicom_dir: Path, output_dir: Path) -> None:
    # TODO(@jguillaumin)
    """

    Parameters
    ----------
    dicom_dir :
    output_dir :

    Returns
    -------

    """
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True)

    logger.info(f"{dicom_dir.as_posix() = }")
    all_tags = []
    for dcm_path in tqdm(dicom_dir.glob("**/*.dcm"), desc="Extract DICOMs to H5 files"):
        # for dcm_path in tqdm(list(dicom_dir.glob("**/*.dcm"))[:5], desc="Extract DICOMs to H5 files"):
        dcm = DCM(dcm_path)
        tags = dcm.tag_dict()
        h5_path = output_dir.absolute() / f"{dcm.image_uid}.h5"

        tags["dcm_path"] = dcm_path.absolute().as_posix()
        tags["h5_path"] = h5_path.as_posix()
        all_tags.append(tags)
        dcm.to_h5(h5_path)

    pd.DataFrame(data=all_tags).to_csv(output_dir / "tags.csv", index=False, header=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dicom_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    params = parser.parse_args(sys.argv[1:])
    dcm_to_h5(dicom_dir=params.dicom_dir, output_dir=params.output_dir)
