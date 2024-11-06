import sys

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from deepjoint_torch.h5 import load_image_meta


def process_inference_results(output_dir: Path, h5_dir: Path):
    # Load image metadata
    image_meta = load_image_meta(h5_dir)

    # Import inference results
    infer_results = pd.read_csv(output_dir / "predictions.csv")

    # Merge inference results with the original dataset
    df_imagelevel = infer_results.merge(
        image_meta[
            [
                "image_uid",
                "patient_id",
                "study_uid",
                "study_date",
                "breast_density_score",
                "view",
                "laterality",
                "status",
                "patient_age",
                "source",
                "harmonized_manufacturer",
            ]
        ],
        on="image_uid",
        how="inner",
        validate="1:1",
    )
    df_imagelevel.sort_values(by=["patient_id", "study_date"], ascending=[True, True], inplace=True)

    # Group data by 'patient_id', 'study_uid', 'laterality' (breast-level) and aggregate 'pred_percent_density' (mean) and 'pred_dense_area' (mean) estimations
    df_breastlevel = (
        df_imagelevel.groupby(["patient_id", "study_uid", "laterality"])
        .agg(
            {
                "pred_percent_density": "mean",
                "pred_dense_area": "mean",
                "pred_breast_area": "mean",
                "study_date": "first",
                "breast_density_score": "first",
                "status": "first",
                "patient_age": "first",
                "source": "first",
                "harmonized_manufacturer": "first",
            }
        )
        .reset_index()
    )
    df_breastlevel.sort_values(by=["patient_id", "study_date"], ascending=[True, True], inplace=True)

    # Group data by 'patient_id', 'study_uid' (visit-level) and aggregate 'pred_percent_density' (mean) and 'pred_dense_area' (sum) estimations
    df_visitlevel = (
        df_breastlevel.groupby(["patient_id", "study_uid"])
        .agg(
            {
                "pred_percent_density": "mean",
                "pred_dense_area": "sum",
                "pred_breast_area": "sum",
                "study_date": "first",
                "breast_density_score": "first",
                "status": "first",
                "patient_age": "first",
                "source": "first",
                "harmonized_manufacturer": "first",
            }
        )
        .reset_index()
    )
    df_visitlevel.sort_values(by=["patient_id", "study_date"], ascending=[True, True], inplace=True)

    # Save visit-level final dataset
    df_visitlevel.to_csv(output_dir / "df_visitlevel.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("-h5", "--h5_dir", dest="h5_dir", type=Path, required=True)

    params = parser.parse_args(sys.argv[1:])

    process_inference_results(output_dir=params.output_dir, h5_dir=params.h5_dir)
