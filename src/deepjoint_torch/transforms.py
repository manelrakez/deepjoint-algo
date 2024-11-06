import cv2
import numpy as np

from loguru import logger
from deepjoint_torch.h5 import DataSample
from deepjoint_torch.constants import IMAGE_WIDTH, IMAGE_HEIGHT


def transforms(sample: DataSample) -> DataSample:
    sample = apply_resize_along_height(sample, target_height=IMAGE_HEIGHT)
    sample = apply_crop_pad_width(sample, target_width=IMAGE_WIDTH)
    sample = apply_voilut(sample)
    sample = apply_breast_masking(sample, background_value=0.0)
    return sample


def apply_photometric_interpretation(sample: DataSample) -> DataSample:
    if sample.photometric_interpretation == "MONOCHROME1":
        logger.debug("apply_photometric_interpretation() with MONOCHROME1")
        min_v = sample.image_min
        max_v = sample.image_max
        sample.image = max_v - sample.image + min_v
        sample.breast_mean = max_v - sample.breast_mean + min_v
        sample.image_min, sample.image_max = sample.image_max, sample.image_min
        sample.breast_min, sample.breast_max = sample.breast_max, sample.breast_min

    return sample


def apply_voilut(sample: DataSample) -> DataSample:
    # scale pixels values in this range
    y_min, y_max = 0.0, 1.0

    voi_lut_function = sample.voi_lut_function
    wl = sample.window_level
    ww = sample.window_width

    def _sigmoid(x):
        # closure on y_min / y_max / wl / ww
        return (y_max - y_min) / (1 + np.exp(-4 * (x - wl) / ww)) + y_min

    def _linear(x):
        # closure on y_min / y_max / wl / ww
        return ((x - wl) / ww + 0.5) * (y_max - y_min) + y_min

    def _transform(_sample, lut_function):
        # closure on wll / ww
        _sample.image = lut_function(_sample.image)
        _sample.breast_mean = lut_function(float(_sample.breast_mean))
        _sample.image_min = lut_function(float(_sample.image_min))
        _sample.image_max = lut_function(float(_sample.image_max))
        _sample.breast_min = lut_function(float(_sample.breast_min))
        _sample.breast_max = lut_function(float(_sample.breast_max))
        return _sample

    # cast array into float32 at this point
    sample.image = sample.image.astype(np.float32)
    if voi_lut_function == "SIGMOID":
        sample = _transform(sample, _sigmoid)
    elif voi_lut_function in ["LINEAR", "LINEAR_EXACT"]:
        sample = _transform(sample, _linear)
    else:
        raise ValueError(f"Unknown VOILUTFunction {voi_lut_function = }")

    sample.image = np.clip(sample.image, y_min, y_max)
    sample.breast_mean = np.clip(sample.breast_mean, y_min, y_max)
    sample.image_min = np.clip(sample.image_min, y_min, y_max)
    sample.image_max = np.clip(sample.image_max, y_min, y_max)
    sample.breast_min = np.clip(sample.breast_min, y_min, y_max)
    sample.breast_max = np.clip(sample.breast_max, y_min, y_max)

    return sample


def apply_breast_masking(sample: DataSample, background_value: float | int | None = None) -> DataSample:
    pixels_outside_breast_mask = sample.breast_mask == 0
    if background_value is None:
        sample.image[pixels_outside_breast_mask] = sample.breast_mean
    else:
        sample.image[pixels_outside_breast_mask] = background_value

    if sample.annotations:
        for mask_name in sample.annotations.keys():
            sample.annotations[mask_name][pixels_outside_breast_mask] = 0

    return sample


def apply_resize_along_height(sample: DataSample, target_height: int = 448) -> DataSample:
    scale_ratio = sample.height / target_height

    interpolation = cv2.INTER_AREA if scale_ratio > 1 else cv2.INTER_LINEAR  # noqa
    new_shape_cv2 = (int(sample.width / scale_ratio), int(sample.height / scale_ratio))  # (W, H) !

    # conversion with uint16 is very important to avoid negative values
    sample.image = cv2.resize(sample.image.astype(np.uint16), new_shape_cv2, interpolation=interpolation)  # noqa
    sample.breast_mask = cv2.resize(sample.breast_mask, new_shape_cv2, interpolation=cv2.INTER_NEAREST)  # noqa

    if sample.annotations:
        for mask_name in sample.annotations.keys():
            mask = sample.annotations[mask_name]
            sample.annotations[mask_name] = cv2.resize(mask, new_shape_cv2, interpolation=cv2.INTER_NEAREST)  # noqa

    return sample


def apply_crop_pad_width(sample: DataSample, target_width: int = 448) -> DataSample:
    diff = target_width - sample.width

    sample.image = _crop_or_pad(sample.image, diff, crop_pad_direction=sample.crop_pad_direction)
    sample.breast_mask = _crop_or_pad(sample.breast_mask, diff, crop_pad_direction=sample.crop_pad_direction)

    if sample.annotations:
        for mask_name in sample.annotations.keys():
            mask = sample.annotations[mask_name]
            sample.annotations[mask_name] = _crop_or_pad(
                mask, diff, crop_pad_direction=sample.crop_pad_direction
            )

    return sample


def _crop_or_pad(image: np.ndarray, crop_pad_value: int, crop_pad_direction: str) -> np.ndarray:
    if crop_pad_value > 0:
        image = _pad_to_width(image, crop_pad_value, crop_pad_direction)
    elif crop_pad_value < 0:
        image = _crop_to_width(image, -crop_pad_value, crop_pad_direction)
    else:
        # force read if image is from hdf5
        image = image[:]
    return image


def _pad_to_width(image: np.ndarray, pad_value: int, crop_pad_direction: str) -> np.ndarray:
    to_pad = [[0, 0]] * image.ndim
    if crop_pad_direction == "L":  # pad to left
        to_pad[-1] = [pad_value, 0]
        pad_value = image[..., 0:10].mean()
    elif crop_pad_direction == "R":  # pad to right
        to_pad[-1] = [0, pad_value]
        pad_value = image[..., -10:-1].mean()
    else:
        raise NotImplemented(f"_pad_to_width() not implemented for CropPadDirection = {crop_pad_direction}")
    image = np.pad(image, to_pad, mode="constant", constant_values=pad_value)  # noqa

    return image


def _crop_to_width(image: np.ndarray, crop_value: int, crop_pad_direction: str) -> np.ndarray:
    if crop_pad_direction == "L":  # crop left
        image = image[..., crop_value : image.shape[1]]  # noqa
    elif crop_pad_direction == "R":
        image = image[..., 0:-crop_value]
    else:
        raise NotImplemented(f"_crop_to_width() not implemented for CropPadDirection = {crop_pad_direction}")
    return image
