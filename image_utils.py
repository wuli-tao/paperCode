import cv2
import numpy as np
from imgaug import augmenters as iaa


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var ** 0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped


def flip_image(image_array):
    return cv2.flip(image_array, 1)


def blur(image_array):
    contrast = iaa.GaussianBlur(sigma=(0.25, 1.25))
    return contrast.augment_image(image_array)


def hue_and_saturation(image_array):
    hue_saturation = iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True)
    return hue_saturation.augment_image(image_array)


def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d


def crop(image_array):
    crop = iaa.CropToFixedSize(width=200, height=200)
    return crop.augment_image(image_array)


def dropout(image_array):
    dropout = iaa.Dropout(p=(0.03, 0.2))
    return dropout.augment_image(image_array)


def sharpen(image_array):
    shrapen = iaa.Sharpen(alpha=(0, 1), lightness=(0.75, 1.5))
    return shrapen.augment_image(image_array)
