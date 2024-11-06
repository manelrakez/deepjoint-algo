from pathlib import Path

import h5py
import numpy as np
import pydicom.multival
from loguru import logger
from pydicom import Dataset, dcmread
from typing import Any
import re
from datetime import datetime
from deepjoint_torch.dcm_utils import (
    fmt_study_date,
    fmt_acquisition_time,
    fmt_manufacturer,
    fmt_patient_age,
    fmt_windowing,
    read_dcm_image,
    generate_mask,
    get_side,
    image_minmax,
    get_flips,
    get_breast_minmax_meanstd,
)

DEFAULT_PS_PATTERN = re.compile(r"NOT_SET|[\(\[]?-1(\.0?)?[\s,]+-1(\.0?)?[\)\]]?")


class DCM:
    __dicom_tags__: set[str] = {
        "patient_id",
        "study_uid",
        "image_uid",
        "study_date",
        "acquisition_time",
        "manufacturer",
        "laterality",
        "view",
        "pres_intent_type",
        "patient_orientation",
        "pixel_spacing",
        "patient_age",
        "photometric_interpretation",
        "window_width",
        "window_level",
        "voi_lut_function",
        "crop_pad_direction",
        "image_height",
        "image_width",
    }

    def __init__(self, dcm_path: Path):
        self.dcm_path = dcm_path

        self.raw_dcm: Dataset = dcmread(dcm_path, stop_before_pixels=False)  # always read pixels
        self._image = None
        self._mask = None

    def __getitem__(self, tag_name: str) -> Any:
        if tag_name not in self.__dict__:
            raise KeyError(f"'{tag_name}' is not a known DICOM tag")
        return getattr(self, tag_name)

    def __str__(self) -> str:
        return f"pydicom.Dataset('{self.dcm_path.as_posix()}')"

    def __repr__(self):
        return str(self)

    def tag_dict(self) -> dict[str, Any]:
        tag_dict = {key: getattr(self, key) for key in self.__dicom_tags__}
        return tag_dict

    def to_h5(self, output_file: Path):
        ps_tuple = tuple(float(elem) for elem in self.pixel_spacing.split(","))
        image = self.image
        mask = self.mask
        crop_pad_direction = self.crop_pad_direction

        flip_hor, flip_ver = get_flips(self.laterality, self.patient_orientation, image_uid=self.image_uid)

        if flip_hor:
            image = np.flip(image, axis=-1)
            mask = np.flip(mask, axis=-1)
            if crop_pad_direction == "L":
                crop_pad_direction = "R"
            elif crop_pad_direction == "R":
                crop_pad_direction = "L"

        if flip_ver:
            image = np.flip(image, axis=-2)
            mask = np.flip(mask, axis=-2)

        breast_min, breast_max, breast_mean, breast_std = get_breast_minmax_meanstd(image, mask)
        image_min, image_max = image_minmax(image)

        data_dict = {
            "photometric_interpretation": (self.photometric_interpretation, h5py.string_dtype()),
            "crop_pad_direction": (crop_pad_direction, h5py.string_dtype()),
            "windowing": ((self.window_level, self.window_width), np.int32),
            "voi_lut_function": (self.voi_lut_function, h5py.string_dtype()),
            "pixel_spacing": (np.asarray(ps_tuple, dtype=np.float32), np.float32),
            "breast_min": (breast_min, image.dtype),
            "breast_max": (breast_max, image.dtype),
            "breast_mean": (breast_mean, np.float32),
            "breast_std": (breast_std, np.float32),
            "image_min": (image_min, image.dtype),
            "image_max": (image_max, image.dtype),
        }

        with h5py.File(output_file, mode="w", libver="latest") as h5_file:
            h5_file.create_dataset("image", data=image, dtype=image.dtype, compression="lzf")
            h5_file.create_dataset("mask", data=mask, dtype=mask.dtype, compression="lzf")

            for name, (data, dtype) in data_dict.items():
                h5_file.create_dataset(name, data=data, dtype=dtype)

    @property
    def patient_id(self) -> str:
        return self.raw_dcm.PatientID

    @property
    def study_uid(self) -> str:
        return self.raw_dcm.StudyInstanceUID

    @property
    def image_uid(self) -> str:
        return self.raw_dcm.SOPInstanceUID

    @property
    def study_date(self) -> datetime:
        return fmt_study_date(self.raw_dcm.StudyDate)

    @property
    def acquisition_time(self) -> str:
        if "AcquisitionTime" in self.raw_dcm:
            return fmt_acquisition_time(self.raw_dcm.AcquisitionTime)
        default = "00:00:00.000000"
        logger.warning(f"No 'AcquisitionTime' for {self}. Use {default = }")
        return default

    @property
    def manufacturer(self) -> str:
        return fmt_manufacturer(self.raw_dcm.Manufacturer)

    @property
    def laterality(self) -> str:
        return self.raw_dcm.ImageLaterality

    @property
    def view(self) -> str:
        code_value = self.raw_dcm.ViewCodeSequence[0].CodeValue
        if code_value in {"R-10242", "399162004", "cranio-caudal"}:
            return "CC"
        if code_value in {"R-10226", "399368009", "medio-lateral-oblique"}:
            return "MLO"

        raise ValueError(f"Invalid {code_value = } for in ViewCodeSequence for {self}")

    @property
    def pres_intent_type(self) -> str:
        return self.raw_dcm.PresentationIntentType

    @property
    def patient_orientation(self) -> str:
        val = self.raw_dcm.PatientOrientation
        return f"{val[0]}/{val[1]}"

    @property
    def pixel_spacing(self) -> str:
        """
        Format a pixel spacing value as a comma-separated string.

        References
        ----------
        * [Imager Pixel Spacing](https://dicom.innolitics.com/ciods/digital-mammography-x-ray-image/dx-detector/00181164)
        * [Pixel Spacing](https://dicom.innolitics.com/ciods/digital-mammography-x-ray-image/dx-detector/00280030)
        """
        if "PixelSpacing" in self.raw_dcm:
            raw_ps = self.raw_dcm.PixelSpacing
        elif "ImagerPixelSpacing" in self.raw_dcm:
            raw_ps = self.raw_dcm.ImagerPixelSpacing
        else:
            raise ValueError(f"'PixelSpacing' for found in {self}")

        ps = None
        if isinstance(raw_ps, list | tuple | np.ndarray | pydicom.multival.MultiValue):
            try:
                ps = tuple(float(val) for val in raw_ps)
                assert len(ps) == 2
                if ps[0] < 0 or ps[1] < 0:
                    ps = get_default_pixel_spacing(self.manufacturer)
            except AssertionError:
                raise ValueError(f"Unexpected pixel_spacing value at row : '{raw_ps}'")
        if isinstance(raw_ps, str):
            if not DEFAULT_PS_PATTERN.search(raw_ps):
                try:
                    enclosures = re.compile(r"[\[\]\(\)\'\"]")
                    str_val = enclosures.sub("", raw_ps)
                    ps = tuple(float(val) for val in str_val.split(","))
                    assert len(ps) == 2
                except (AssertionError, ValueError):
                    raise ValueError(
                        f"Unexpected pixel_spacing value at row : '{raw_ps = }' - '{str_val = }'"
                    )

        if ps is None:
            raise ValueError(f"Unexpected pixel_spacing value at row : '{raw_ps = }' ({type(raw_ps) = })")

        return ",".join(str(x) for x in ps)

    @property
    def patient_age(self) -> int:
        val = self.raw_dcm.PatientAge
        return fmt_patient_age(val)

    @property
    def photometric_interpretation(self) -> str:
        return self.raw_dcm.PhotometricInterpretation

    @property
    def window_width(self):
        return fmt_windowing(self.raw_dcm.WindowWidth)

    @property
    def window_level(self):
        return fmt_windowing(self.raw_dcm.WindowCenter)

    @property
    def voi_lut_function(self) -> str:
        # VOILUTFunction
        if "VOILUTFunction" in self.raw_dcm:
            val = self.raw_dcm.VOILUTFunction
        elif "SharedFunctionalGroupsSequence" in self.raw_dcm:
            val = (
                self.raw_dcm.SharedFunctionalGroupsSequence[0]
                .FrameVOILUTSequence[0]
                .VOILUTFunction.decode("utf-8")
            )
        else:
            val = "LINEAR"
            logger.warning(f"Use default {val = } for 'VOILUTFunction' in {self}")

        if val not in {"LINEAR", "SIGMOID"}:
            raise ValueError(f"Unknown 'VOILUTFunction' {val = } in {self}")

        return val

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            self._image = read_dcm_image(self.raw_dcm)

        return self._image

    @property
    def image_height(self) -> int:
        return self.image.shape[0]

    @property
    def image_width(self) -> int:
        return self.image.shape[1]

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            self._mask = generate_mask(
                image=self.image,
                manufacturer=self.manufacturer,
                patient_orientation=self.patient_orientation,
                photometric_interpretation=self.photometric_interpretation,
                image_uid=self.image_uid,
            )
        return self._mask

    @property
    def crop_pad_direction(self) -> str:
        side = get_side(
            self.image,
            patient_orientation=self.patient_orientation,
            photometric_interpretation=self.photometric_interpretation,
            image_uid=self.image_uid,
        )
        return "R" if side == "L" else "L"


def get_default_pixel_spacing(harmonized_manufacturer: str, image_type: str) -> tuple[float, float]:
    """
    Get the default pixel spacing for a DICOM.

    This is `(0.07, 0.07)` for Hologic FFDM and `(0.1, 0.1)` for the rest.
    """

    if harmonized_manufacturer == "hologic" and image_type == "mammography":
        return 0.07, 0.07

    return 0.1, 0.1
