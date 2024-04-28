import cv2
import numpy as np
import elasticdeform
from PIL import Image

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

def to_numpy(input_path):
    with Image.open(input_path) as img:
        jpeg_image = img.convert("RGB")
        numpy_array = np.array(jpeg_image)
    return numpy_array

def downsample(image, target_size):
    return cv2.resize(image, target_size)

def apply_deformation(image, displacement):
    # if displacement:
    deformation = elasticdeform.deform_grid(image, displacement, axis=(0, 1))
    # else:
    #     deformation = elasticdeform.deform_random_grid(image, sigma=25, points=3, axis=(0, 1))
    return deformation.astype(np.uint8)

def generate_pairs(original_images, segmentation_masks, num_pairs, target_size, transformation_type):
    pairs = []

    if transformation_type == 'torchvision':
        elastic_transformer = v2.ElasticTransform(alpha=150.0, interpolation=InterpolationMode.NEAREST)

    for i in range(len(original_images)):
        original_img = original_images[i]
        segmentation_mask = segmentation_masks[i]

        downsampled_target_image = downsample(original_img, target_size)
        downsampled_target_mask = downsample(segmentation_mask, target_size)

        if transformation_type == 'manual':
            downsampled_target_image = cv2.cvtColor(downsampled_target_image, cv2.COLOR_RGB2GRAY)
            downsampled_target_mask = cv2.cvtColor(downsampled_target_mask, cv2.COLOR_RGB2GRAY)
                
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_RGB2GRAY)

        downsampled_target_mask = (downsampled_target_mask > 127).view(np.uint8) * 255

        source_masks = []
        source_images = []
        
        for _ in range(num_pairs):

            if transformation_type == 'elasticdeform':
                displacement = np.random.randn(2, 3, 3) * 25

                deformed_mask = apply_deformation(segmentation_mask, displacement)
                deformed_image = apply_deformation(original_img, displacement)

            elif transformation_type == 'torchvision':
                deformed_mask = elastic_transformer(Image.fromarray(segmentation_mask))
                deformed_image = elastic_transformer(Image.fromarray(original_img))

            elif transformation_type == 'manual':
                im_merge = np.concatenate((original_img[...,None], segmentation_mask[...,None]), axis=2)
                transformed = manual_elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.08)

                deformed_image = transformed[...,0]
                deformed_mask = transformed[...,1]
                
            deformed_mask = downsample(deformed_mask, target_size)
            deformed_image = downsample(deformed_image, target_size)

            deformed_mask = (deformed_mask > 127).view(np.uint8) * 255
            
            source_masks.append(deformed_mask)
            source_images.append(deformed_image)

        pairs_dict = {
            'target_image': downsampled_target_image, 
            'target_mask': downsampled_target_mask, 
            'source_images': source_images,
            'source_masks': source_masks
            }

        pairs.append(pairs_dict)
    
    return pairs

def manual_elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def save_image(image, filename):
    pil_image = Image.fromarray(image)
    pil_image.save(filename)
