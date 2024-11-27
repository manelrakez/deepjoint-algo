# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

import numpy as np
from pydicom import Dataset
from datetime import datetime
from typing import Any
from pydicom.multival import MultiValue
from pydicom.pixel_data_handlers.pylibjpeg_handler import generate_frames
import cv2
from skimage import filters
import scipy
from loguru import logger


def fmt_study_date(val: str) -> datetime:
    """
    Parse a date in DICOM string format and return a datetime object.

    Strings can come in the following formats: YYYYMMDD (date), or
    YYYYMMDDHHMMSS.FFFFFF&ZZXX (datetime) -as specified by the DICOM
    standard.

    Parameters
    ----------
    val : str
        A date(time) in DICOM string format.

    Returns
    -------
    datetime
        A datetime corresponding to the DICOM string
        value.

    Raises
    ------
    ValueError
        A ValueError is raised if the dcm_date parameter
        does not conform to any DICOM date(time) format.

    References
    ----------
    [DICOM date VRs](https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html)
    """
    fmt = "%Y%m%d"
    try:
        return datetime.strptime(val, fmt)
    except ValueError:
        fmt += "%H%M%S.%f"
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            fmt += "%z"
            try:
                return datetime.strptime(val, fmt)
            except ValueError:
                # Try our custom date format.
                fmt = "%Y-%m-%d"
                return datetime.strptime(val, fmt)


def fmt_acquisition_time(val: int | float | str) -> str:
    """
    Format the DICOM acquisition time in the 'HH:MM:SS.ffffff' format.

    References
    ----------
    https://dicom.innolitics.com/ciods/digital-mammography-x-ray-image/general-image/00080032
    """
    try:
        return datetime.strptime(str(val), "%H%M%S.%f").strftime("%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(str(val), "%H%M%S").strftime("%H:%M:%S.%f")


def fmt_manufacturer(val: str) -> str:
    if val in {"GE HEALTHCARE", "GE MEDICAL SYSTEMS", "GE Healthcare"}:
        return "ge"
    if val in {"LORAD", "Lorad, A Hologic Company", "HOLOGIC, Inc.", "HOLOGIC", "Hologic, Inc."}:
        return "hologic"

    raise ValueError(f"Unknown manufacturer = {val}")


def fmt_patient_age(val: str):
    """
    Format the patient's age from a DICOM age string to an int.

    Values correspond to the patient's age in years.

    If the age parameter is a null or empty value, or is not correctly
    formatted (ie: nnn[DWMY]), -1 is returned. The only exception to this
    is if the age_str value can be directly coerced to an int.

    Parameters
    ----------
    val : str
        The patient's age as a DICOM age string.

    Returns
    -------
    int
        The patient's age in years.

    Raises
    ------
    ValueError
        An error is raised if the time unit is not recognized.

    References
    ----------
    [Age String VR](https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html)

    Notes
    -----
    Datetime units as strings can be somewhat iffy. The assumptions
    made are that there are 12 months in a year, 52 weeks in a year,
    and 365 days in a year.
    """
    try:
        return int(val)
    except ValueError:
        pass

    if not val:
        return -1

    # Ages should normally be 4 characters long (\d\d\d[DWMY])
    # but sometimes people are lazy and don't 0-pad values. Therefore
    # we'll allow it as long as the unit is known.
    age_str = str(val)
    age_number = int(age_str[:-1])
    age_unit = age_str[-1].upper()
    if age_unit not in "YMWD":
        raise ValueError(
            f"Expected the age string unit to be one of 'D', 'W', 'M', 'Y'. Obtained: {age_unit}"
        )

    if age_unit == "Y":
        return age_number
    if age_unit == "M":
        return int(age_number / 12)
    if age_unit == "W":
        return int(age_number / 52)
    return int(age_number / 365)


def fmt_windowing(val: Any) -> int:
    """
    Return the DICOM window level as an integer.

    References
    ----------
    * [Window Center](https://dicom.innolitics.com/ciods/digital-mammography-x-ray-image/dx-image/00281050)
    * [Window Width](https://dicom.innolitics.com/ciods/digital-mammography-x-ray-image/dx-image/00281051)

    Notes
    -----
    In some cases, the window level/center values can be a list of values
    (eg: for IMS Giotto DBT). In such cases, we systematically pick the first
    value of the list.
    """
    if isinstance(val, list | tuple | MultiValue):
        val = val[0]
    return int(round(float(val)))


def read_dcm_image(dcm: Dataset) -> np.ndarray:
    """
    Read the input DICOM dataset or DICOM file's image array.

    Parameters
    ----------
    dcm : pydicom.Dataset

    Returns
    -------
    np.ndarray
        The DICOM's pixel data as a np.uint16 encoded numpy array.
    """

    image_uid = dcm.SOPInstanceUID
    transfer_syntax = dcm.file_meta.TransferSyntaxUID

    # JPEG2000 and JPEG2000_LOSSLESS
    if transfer_syntax in ["1.2.840.10008.1.2.4.91", "1.2.840.10008.1.2.4.90"]:
        if dcm.BitsStored != 16:
            dcm.BitsStored = 16
            logger.warning(f"{image_uid}: Forcing BitsStored to 16 on a JPEG2000 compressed image")
        # Avoid using the Pillow reader at all costs so explicitly use the pylibjpeg
        # readers instead of a `pixel_array` call.
        frames = list(generate_frames(dcm))
        array = frames[0] if len(frames) == 1 else np.asarray(frames)
    else:
        array = dcm.pixel_array

    dtype = array.dtype
    min_val, max_val = np.min(array), np.max(array)

    if dtype not in [np.uint16, np.int16]:
        logger.warning(f"{image_uid}: DICOM dtype of image array not in [np.uint16, np.int16]: {dtype}")
    if min_val < 0:
        logger.warning(f"{image_uid}: np.int16 DICOM array has a negative minimum of {min_val}")

    if "BitsStored" in dcm:
        bits_stored = int(dcm.BitsStored)
        pixel_repr = int(dcm.PixelRepresentation)

        repr_0_max_val = 2**bits_stored - 1
        repr_1_max_val = 2 ** (bits_stored - 1) - 1
        # pixel_repr == 0 or pixel_repr == 1
        expected_max_val = repr_0_max_val if pixel_repr == 0 else repr_1_max_val

        if max_val > expected_max_val:
            # The 2 scenarios that can happen are:
            # 1 - the array "simply" needs to be byteswapped
            # 2 - the array has some outlier pixels that need to be clipped.
            # To determine which is which, we must first byteswap and then
            # check whether the byteswapped array has a max value that is in
            # the expected range. If this is the case, we are in scenario 1.
            # Otherwise, we are in scenario 2.
            byteswapped_array = array.byteswap()
            bs_max = byteswapped_array.max()
            if bs_max < max_val and bs_max <= expected_max_val:
                logger.warning(f"{image_uid}: Byte-swapping pixel array")
                array = byteswapped_array
            else:
                logger.warning(
                    f"{image_uid}: Clipping value(s) greater than {expected_max_val}"
                    f" to {expected_max_val} (eg: {max_val})"
                )
                array.clip(min_val, expected_max_val, out=array, dtype=array.dtype)

    return array if array.dtype == np.uint16 else array.astype(np.uint16)


def generate_mask(
    image: np.ndarray,
    manufacturer: str,
    patient_orientation: str,
    photometric_interpretation: str,
    image_uid: str,
) -> np.ndarray:
    if np.all(image == image[0, 0]):
        logger.warning(f"{image_uid = } : Empty image detected. Returning an empty mask.")
        return np.zeros(image.shape, dtype=np.uint8)

    if manufacturer == "hologic":
        threshold_value = filters.threshold_triangle(image, nbins=256)
    elif manufacturer == "ge":
        threshold_value = threshold_histogram(image)
    else:
        raise ValueError(f"Invalid {manufacturer = } for {image_uid = }")

    breast_mask = _create_raw_mask(image, photometric_interpretation, threshold_value)

    # reconnect elements closed to each others with a small closing operation
    # permits to reconnect the nipple to the breast in certain cases.
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, disk)

    # select side where breast resides (left or right)
    side = get_side(image, patient_orientation, photometric_interpretation, image_uid=image_uid)
    # extract largest connected component.
    # Permits to remove large bars in the image
    # cv2 documentation:
    # 'Operation mask should be a single-channel 8-bit image,
    # 2 pixels wider and 2 pixels taller than image'

    seed_point = None
    if side in ["L", "R"]:
        ind = np.argwhere(breast_mask[breast_mask.shape[0] // 2, :])
        if list(ind):
            if side == "L":
                # floodfilling from the middle extreme left point
                ind_val = int(ind[0] if not isinstance(ind[0], np.ndarray) else ind[0][0])
                seed_point = (ind_val, breast_mask.shape[0] // 2)
            else:
                # side == 'R'
                # right-most non-negative middle mask position
                ind_val = int(ind[-1] if not isinstance(ind[-1], np.ndarray) else ind[-1][0])
                seed_point = (ind_val, breast_mask.shape[0] // 2)

    if seed_point:
        new_breast_mask = np.zeros(tuple(x + 2 for x in breast_mask.shape), dtype=np.uint8)
        cv2.floodFill(
            image=breast_mask,
            mask=new_breast_mask,
            seedPoint=seed_point,
            newVal=255,
            flags=cv2.FLOODFILL_MASK_ONLY,
        )
        # back to original shape
        breast_mask = new_breast_mask[1:-1, 1:-1]
    else:
        # conversion into uint8
        logger.warning(f"{image_uid = } : Flood fill method cannot be applied due to missing seed point.")

    # remove smaller (noisy) elements with disk of radius 11
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, disk)
    # fill holes and smooth contours with larger disk
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, disk)

    return breast_mask


def threshold_histogram(image: np.ndarray) -> float:
    """
    Custom threshold mechanism, which selects a threshold value based on the image histogram.

    It has been created to segment GE images.
    """
    nbins = 256

    amin, amax = image_minmax(image)
    hist, _ = image_histogram(image, nbins=nbins, array_minmax=(int(amin), int(amax)))
    # Get the two most represented histogram bin
    hist_max_count_idx = np.argmax(hist)

    offset = hist_max_count_idx + 30  # I choose 30 because it's bigger than the maximun width of a peak
    if offset >= nbins:
        offset = hist_max_count_idx
    second_hist_max_count_idx = np.argmax(hist[int(offset) :]) + offset

    # we have to add a 0 at the begining and ending of the histogram so that
    # it can found peak at the begining and the end
    peaks, properties = scipy.signal.find_peaks(
        [0, *list(hist), 0], height=max(hist) // 5, distance=20, width=1
    )
    peaks = peaks - 1

    nb_peak = len(peaks)
    max_peak_width = max(properties["widths"])
    max_count_at_0 = hist_max_count_idx == 0
    more_than_2_peaks = nb_peak > 2
    less_than_2_peaks = nb_peak < 2
    there_are_2_peaks = nb_peak == 2
    peak_width_under_10 = max_peak_width < 10
    peak_width_under_2 = max_peak_width < 2.1
    peak_width_different_to_1 = max_peak_width != 1
    second_peak_not_at_0 = min(peaks) != 0
    peak_not_at_last_bin = max(peaks) != nbins - 1
    second_max_count_at_first_third = second_hist_max_count_idx < nbins // 3
    max_count_at_second_half = hist_max_count_idx > nbins // 2

    # image with weird gray background
    if more_than_2_peaks and max_count_at_0 and peak_width_under_10:
        thresh = amax * 0.6

    elif (
        (
            there_are_2_peaks
            and not max_count_at_0
            and second_peak_not_at_0
            and peak_width_under_2
            and peak_not_at_last_bin
        )
        or (
            less_than_2_peaks
            and max_count_at_0
            and second_max_count_at_first_third
            and peak_width_different_to_1
        )
        or (less_than_2_peaks and not max_count_at_0 and max_count_at_second_half)
    ):
        thresh = filters.threshold_triangle(image, nbins=256)

    # black and gray rectangle background, big implant, large breast and other
    else:
        thresh = threshold_li_high_guess(image)

    return thresh


def threshold_li_high_guess(image: np.ndarray) -> float:
    """
    Compute the image threshold value using the li algorithm.

    Initialize guesses using a semi-arbitrary high value from the
    array. This is designed for GE images because some of them may
    have black and gray values. This causes the threshold algorithm
    to find a local minima between the black and gray pixel group
    (instead of between the gray and breast pixel group).

    See `skimage.filters.thresholding:threshold_li` for more details.
    """
    image_max, image_mean = get_breast_minmax_meanstd(image)[1:3]
    # The initial guess will be between the mean and the max value.
    high_guess = (image_max + image_mean) // 2
    return filters.threshold_li(image, initial_guess=high_guess)


def image_minmax(image: np.ndarray) -> tuple[int | float, int | float]:
    """Return a 2D image's min and max values."""
    # This is faster than doing numpy.min(arr) and numpy.max(arr)
    amin, amax = cv2.minMaxLoc(image)[:2]
    if image.dtype.kind in "iu":
        amin, amax = int(amin), int(amax)
    return amin, amax


def image_mean_std(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """
    Return a 2D or 3D image's mean and stddev.

    If mask is provided, only pixels of `image` where `mask` is True
    will be considered for the computation of the mean and stddev.
    """
    if mask is not None:
        mask = mask.ravel()
    mean_arr, std_arr = cv2.meanStdDev(image.ravel(), mask=mask)
    return mean_arr.ravel()[0], std_arr.ravel()[0]


def count_non_zero(image: np.ndarray) -> int:
    """Return the non-zero pixel count in the image array."""
    if image.ndim == 2:
        return cv2.countNonZero(image)
    return sum(cv2.countNonZero(im_slice) for im_slice in image)


def image_histogram(
    array: np.ndarray,
    mask: np.ndarray | None = None,
    nbins: int = 256,
    array_minmax: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the array's histogram and associated bins.

    Equivalent (but faster, especially for 3D) to:
    ```
    hist, bins = np.histogram(array, bins=256)
    ```

    If the mask is given, this is equivalent to:
    ```
    hist, bins = np.histogram(array[mask == 1], bins=256)
    ```

    Parameters
    ----------
    array : np.ndarray
        The input array to obtain a histogram for.
    mask : np.ndarray | None
        If specified, compute the histogram only for the pixels
        inside the mask.
    nbins : int
        The number of bins to produce in the histogram.
    array_minmax : tuple[int, int] | None
        If set, specify

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A 2-tuple consisting of the image array's histogram values
        and bins.
    """
    arrmin, arrmax = array_minmax or image_minmax(array)

    hist_range = [arrmin, arrmax + 1]
    bins = np.linspace(arrmin, arrmax, nbins + 1, endpoint=True, dtype=array.dtype)

    if array.ndim == 2:
        hist = cv2.calcHist([array], [0], mask, [nbins], hist_range).ravel()
    else:
        mask_list: np.ndarray | list[None] = mask if mask is not None else [None] * len(array)
        hist_arr = np.array(
            [
                cv2.calcHist([array[idx]], [0], mask_list[idx], [nbins], hist_range).ravel()
                for idx in range(len(array))
            ]
        )
        hist = np.asarray(hist_arr.sum(axis=0))

    hist = hist.astype(np.int64)
    # Slightly odd behavior from cv2 where the first bin can miss a single count.
    if hist.sum() == array.size - 1:
        hist[0] += 1

    return hist, bins


def get_breast_minmax_meanstd(
    array: np.ndarray,
    mask: np.ndarray | None = None,
    nbins: int = 512,
    array_minmax: tuple[int, int] | None = None,
) -> tuple[int, int, float, float]:
    """
    Compute the breast min, max, mean, and stddev values.

    Values are obtained based on a histogram approach. Min
    and max values are computed as the top Nth (2.5% or 10% if DBT
    and 97.5% or 90% if DBT) values.

    Parameters
    ----------
    array : np.ndarray
        The input image array.
    mask : np.ndarray | None
        The image's associated breast mask array, if available.
        If provided, min, max, mean, and std values will only be
        computed for pixels inside this mask.
    nbins : int
        The number of bins to
    array_minmax : tuple[int, int] | None
        If provided, specify the image's min and max values.
    Returns
    -------
    tuple[int, int, float, float]
        The breast's min, max, mean, and stddev values, respectively.
    """
    hist, hist_bins = image_histogram(array, mask=mask, nbins=nbins, array_minmax=array_minmax)
    hist_centers = (hist_bins[:-1] + hist_bins[1:]) / 2.0

    hist_sum = hist.sum()
    hist_cumsum_arr = np.cumsum(hist)

    # For 2D: take the bottom 2.5% and top 97.5%
    # For 3D: take the bottom 10% and top 90%
    bottom, top = (0.025, 0.975) if array.ndim == 2 else (0.1, 0.9)
    # Start by retrieving the greatest value smaller or equal to our target value.
    # In some edge cases, your max candidate can be smaller than our
    bmin_candidates = np.where(hist_cumsum_arr <= (hist_sum * bottom))[0]
    bmin_idx = bmin_candidates.max() if len(bmin_candidates) > 0 else 0
    bmin = hist_centers[bmin_idx]

    bmax_candidates = np.where(hist_cumsum_arr <= (hist_sum * top))[0]
    bmax_idx = bmax_candidates.max() if len(bmax_candidates) > 0 else len(hist_centers) - 1
    bmax = hist_centers[bmax_idx]

    bmean = np.average(hist_centers, weights=hist)
    bstd = np.sqrt(np.average((hist_centers - bmean) ** 2, weights=hist))

    return int(bmin), int(bmax), bmean, bstd


def _create_raw_mask(
    image_array: np.ndarray, photometric_interpretation: str, threshold_value: float
) -> np.ndarray:
    """
    Create the mask of the breast based on the photometric interpretation and the threshold value.

    Parameters
    ----------
    image_array : np.ndarray
        The DICOM image's pixel array.
    photometric_interpretation : str
        The DICOM value of tag PhotometricInterpretation ('MONOCHROME2' or 'MONOCHROME1')
    threshold_value : float
        The threshold value used to segment the breast.

    Returns
    -------
    np.ndarray
        The mask of the breast
    """
    if photometric_interpretation == "MONOCHROME1":
        breast_mask = image_array < threshold_value  # type: np.ndarray
    else:
        breast_mask = image_array > threshold_value
    return breast_mask.astype(np.uint8)


def get_side(
    image: np.ndarray,
    patient_orientation: str,
    photometric_interpretation: str,
    image_uid: str = "",
) -> str:
    """
    Determine the side of the image in which the breast resides.

    Parameters
    ----------
    image : np.ndarray
        The DICOM image's pixel array.
    patient_orientation : str
        The DICOM value of tag PatientOrientation as a list of strings. Ex: ['A', 'P'].
    photometric_interpretation : str
        The DICOM value of tag PhotometricInterpretation ('MONOCHROME2' or 'MONOCHROME1')
    image_uid : str
        An optional identifier to prepend warning messages with.

    Returns
    -------
    str
        'L' (breast aligned left) or 'R' (breast aligned right)
    """
    side = None
    if patient_orientation is not None:
        if "A" in patient_orientation:
            side = "L"
        elif "P" in patient_orientation:
            side = "R"
        else:
            logger.warning(f"{image_uid = } : Unsupported patient orientation: {patient_orientation}")

    if side is not None:
        return side

    logger.warning(
        f"{image_uid = } : defaulting to L/R band technique to determine side where breast resides"
    )
    # no need to assume whether we have a tomo or a mammo provided width is the last dimension
    left_band_mean = image[..., :10].mean()
    right_band_mean = image[..., -10:].mean()

    if photometric_interpretation == "MONOCHROME1":
        # The minimum sample value is intended to be displayed as white (ie breast has a low value)
        side = "L" if left_band_mean < right_band_mean else "R"
    elif photometric_interpretation == "MONOCHROME2":
        # The minimum sample value is intended to be displayed as black (ie breast has a high value)
        side = "L" if left_band_mean > right_band_mean else "R"
    else:
        raise ValueError(
            f"{image_uid = } : PhotometricInterpretation='{photometric_interpretation}'"
            " is not supported for breast side determination."
        )
    return side


def get_flips(laterality: str, patient_orientation: str, image_uid: str) -> tuple[bool, bool]:
    """
    Determine if a DBT or a 2DSM image shall be flipped horizontally or vertically.

    It then returns recommended flips (horizontal, vertical).

    Parameters
    ----------
    laterality : str
        The anatomical laterality.
    patient_orientation : str
        The orientation for the patient_orientation.
    image_uid : str
        An optional log identifier to prepend warning logs with.

    Returns
    -------
    tuple[bool, bool]
        A 2-tuple whose first element indicates whether
        to flip the image horizontally and whose second
        element indicates whether to flip the image
        vertically.
    """
    flip_hor, flip_ver = False, False

    try:
        hpo, vpo = str(patient_orientation).split("/")
    except ValueError:
        logger.info(f"{image_uid = } : unset or wrongly formatted patient orientation: {patient_orientation}")
        return flip_hor, flip_ver

    # Standard patient orientation:
    # - RCC: P/L
    # - RMLO: P/F
    # - LCC: A/R
    # - LMO: A/F
    flip_hor = (laterality == "R" and "A" in hpo) or (laterality == "L" and "P" in hpo)
    flip_ver = ("H" in vpo) or (laterality == "R" and "R" in vpo) or (laterality == "L" and "L" in vpo)

    return flip_hor, flip_ver
