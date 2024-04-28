import numpy as np
from PIL import Image
import cv2

def convert_to_uint(array_values, threshold=0.0):
    binary_array = (array_values > threshold).astype(np.uint8)
    uint8_array = binary_array * 255
    return uint8_array

def convert_to_uint_v2(array_values):
    array_values = np.clip(array_values, 0, 255)
    uint8_array = array_values.astype(np.uint8)
    return uint8_array


def load_color_image(image_path, resize=None):
    pil_image = Image.open(image_path)
    numpy_array = np.array(pil_image)

    if resize:
        resized = cv2.resize(numpy_array, resize)
        return resized

    return numpy_array

def load_gray_image(image_path, resize=None):
    color_image = load_color_image(image_path, resize)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    return gray_image