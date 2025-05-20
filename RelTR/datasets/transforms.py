# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import math
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate
import numpy as np


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    

### BELOW ARE SELF IMPLEMENTED TRANSFORMATIONS ###

class RandomColorColumnsPadding(object):
    """
    Append a random number (between min_cols and max_cols) of single-pixel-wide
    columns of random color to a random side of the image.
    Then resize to a target size.

    USE BEFORE TO TENSOR
    """
    def __init__(self, min_cols: int, max_cols: int, size, p: float = 0.3):
        self.min_cols = min_cols
        self.max_cols = max_cols
        self.size = size
        self.p = p


    def __call__(self, img: Image.Image, target):
        if random.random() > self.p:
            return F.resize(img, self.size), target

        direction = random.choice(['left', 'right', 'top', 'bottom'])
        w, h = img.size
        n_cols = random.randint(self.min_cols, self.max_cols)

        img_np = np.array(img)
        mode = img.mode

        if mode == 'RGB':
            c = 3
        elif mode == 'L':
            c = 1
            img_np = img_np[:, :, None]  # Add channel dimension
        else:
            raise ValueError(f"Unsupported image mode: {mode}")

        # Shape for the random color padding block
        if direction in ['left', 'right']:
            pad_shape = (h, n_cols, c)
        else:  # top or bottom
            pad_shape = (n_cols, w, c)

        # Random colors for the padding
        pad_array = np.random.randint(0, 256, size=pad_shape, dtype=np.uint8)

        # Concatenate along correct axis
        if direction == 'left':
            new_img = np.concatenate((pad_array, img_np), axis=1)
        elif direction == 'right':
            new_img = np.concatenate((img_np, pad_array), axis=1)
        elif direction == 'top':
            new_img = np.concatenate((pad_array, img_np), axis=0)
        else:  # bottom
            new_img = np.concatenate((img_np, pad_array), axis=0)

        if c == 1:
            new_img = new_img[:, :, 0]  # Remove channel axis for grayscale

        padded_img = Image.fromarray(new_img, mode=mode)

         # Adjust bounding boxes
        if target is not None and "boxes" in target:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.clone()
            else:
                boxes = torch.tensor(boxes).clone()

            if direction == 'left':
                boxes[:, [0, 2]] += n_cols
            elif direction == 'right':
                pass  # no shift needed
            elif direction == 'top':
                boxes[:, [1, 3]] += n_cols
            elif direction == 'bottom':
                pass  # no shift needed

            target["boxes"] = boxes

        #return resize(padded_img, target, self.size)
        return padded_img, target

    def __repr__(self):
        return (f"{self.__class__.__name__}(min_cols={self.min_cols}, "
                f"max_cols={self.max_cols}, size={self.size}, p={self.p})")
    



class SparseColorNoise(object):
    """
    Randomly modifies up to `max_pct` percent of pixels in an image,
    distributing them as far apart as possible.
    """

    def __init__(self, max_pct: float = 0.05, p : float = 0.3):
        assert 0.0 < max_pct <= 1.0, 'max_pct must be between 0 and 1'
        self.max_pct = max_pct
        self.p = p


    def __call__(self, img: Image.Image, target):
        if random.random() > self.p:
            return img, target

        img_np = np.array(img)
        h, w = img_np.shape[:2]
        num_pixels = h * w
        num_changes = int(num_pixels * self.max_pct)

        if num_changes == 0:
            return img, target

        # Generate unique random indices
        flat_indices = np.random.choice(num_pixels, size=num_changes, replace=False)
        ys, xs = np.unravel_index(flat_indices, (h, w))

        # Generate random RGB or grayscale values
        if img.mode == 'RGB':
            random_colors = np.random.randint(0, 256, size=(num_changes, 3), dtype=np.uint8)
            img_np[ys, xs] = random_colors
        elif img.mode == 'L':
            random_colors = np.random.randint(0, 256, size=num_changes, dtype=np.uint8)
            img_np[ys, xs] = random_colors
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")

        return Image.fromarray(img_np, mode=img.mode), target

    def __repr__(self):
        return f"{self.__class__.__name__}(max_pct={self.max_pct})"
    

class RandomGrayscale(object):
    def __init__(self, p=0.3):
        assert 0.0 < p <= 1.0, 'p must be between 0 and 1'
        self.p = p
        self.grayscale_transform = T.RandomGrayscale(p)  # always convert if called

    def __call__(self, image, target):  
        return self.grayscale_transform(image), target
    

class RandomAdjustSharpness(object):
    def __init__(self, sharpness_factor=2, p=0.3):
        assert 0.0 < p <= 1.0, 'p must be between 0 and 1'
        self.p = p
        self.sharpness_factor = sharpness_factor
        self.sharpness_transform = T.RandomAdjustSharpness(sharpness_factor, p)  # always convert if called

    def __call__(self, image, target):
        return self.sharpness_transform(image), target


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor=2, p=0.3):
        assert 0.0 < p <= 1.0, 'p must be between 0 and 1'
        self.p = p
        self.sharpness_factor = sharpness_factor
        self.transform = T.RandomAdjustSharpness(sharpness_factor, p=1.0)  # apply always when called

    def __call__(self, image, target):
        if random.random() > self.p:
            return image, target
        return self.transform(image), target


class RandomGaussianBlur:
    def __init__(self, kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3):
        assert 0.0 < p <= 1.0, 'p must be between 0 and 1'
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.transform = T.GaussianBlur(kernel_size, sigma=sigma)

    def __call__(self, image, target):
        if random.random() > self.p:
            return image, target
        return self.transform(image), target


class RandomColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.3):
        assert 0.0 < p <= 1.0, 'p must be between 0 and 1'
        self.p = p
        self.transform = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, target):
        if random.random() > self.p:
            return image, target
        return self.transform(image), target


class RandomAutocontrast:
    def __init__(self, p=0.3):
        assert 0.0 < p <= 1.0, 'p must be between 0 and 1'
        self.p = p
        self.transform = T.RandomAutocontrast(p=1.0)  # torchvision accepts p here, but to be safe handle manually

    def __call__(self, image, target):
        return self.transform(image), target