import numpy as np
import tensorflow as tf

def generate_data_triangles(data_size=128, hollow=False, samples=6000):
    '''
    '''
    x, y = np.mgrid[0 : 1 : data_size * 1j, 0 : 1 : data_size * 1j]
    x = np.reshape(x, (1, data_size, data_size))
    y = np.reshape(y, (1, data_size, data_size))
    cx = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    cy = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    r = np.random.random((samples, 1, 1)) * 0.2 + 0.2
    theta = np.random.random((samples, 1, 1)) * np.pi * 2
    isTriangle = np.random.random((samples, 1, 1)) > 0.5

    triangles = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r * np.cos(np.pi / 3) / np.cos(
        (np.arctan2(x - cx, y - cy) + theta) % (2 * np.pi / 3) - np.pi / 3
    )

    triangles = np.tanh(-40 * triangles)

    circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r))
    if hollow:
        triangles = 1 - triangles**2
        circles = 1 - circles**2

    images = isTriangle * triangles + (1 - isTriangle) * circles

    return images

def data_triangles_split(data_size=128, hollow=False, samples=6000):
    '''
    '''
    images = generate_data_triangles(data_size, hollow, samples)

    np.random.shuffle(images)

    split_index = len(images) // 2

    array1 = images[:split_index]
    array2 = images[split_index:]

    return array1, array2