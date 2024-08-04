
import numpy as np
from matplotlib import pyplot
from PIL import Image
from typing import Any
from src.logging import get_logger


log = get_logger('utils.py')

def convert_str_to_array(string: str):
    return np.array(list(map(int, string.split())))

def convert_1D_array_to_image(array: np.ndarray, height: int, width: int):
    if array.size == height * width:
        return array.reshape((height, width))
    else:
        raise ValueError(f"The size of the vector {array.size} does not correspond to the product of the dimensions ({height}, {width})")

def write_image(image: np.ndarray[Any, Any], output: str, cmap: str = 'gray'):
    pyplot.imsave(output, image, cmap=cmap)
    log.debug(f'Writing image to {output}')

def load_image(image_path: str):
    log.debug(f'Loading image {image_path}')
    image = Image.open(image_path).convert('L')  # Load grayscale image
    return np.array(image)
