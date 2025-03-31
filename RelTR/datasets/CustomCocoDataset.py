import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import os
import numpy as np
import random
import json
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
#import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image


class Preparator():
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.dataset_size = len(self.image_ids)

    def split_train_val_test(self, split):
        # Compute the number of samples for each split
        number_train_data = int(self.dataset_size * split[0])
        number_validation_data = int(self.dataset_size * split[1])
        
        # Randomly select images for the training set
        draws_train = random.sample(self.image_ids, number_train_data)
        
        # Remaining images after selecting training set
        remaining_images = [value for value in self.image_ids if value not in draws_train]
        
        # Randomly select images for the validation set from the remaining images
        draws_validation = random.sample(remaining_images, number_validation_data)
        
        # The remaining images after selecting validation set will be the test set
        draws_test = [value for value in remaining_images if value not in draws_validation]
        
        return draws_train, draws_validation, draws_test
    
    def create_split_annotations(self, split_ids, output_file):
        images = [self.coco.imgs[img_id] for img_id in split_ids]
        annotations = [ann for ann in self.coco.anns.values() if ann['image_id'] in split_ids]
        categories = list(self.coco.cats.values())
        split_data = {
            "images" : images,
            "annotations" : annotations,
            "categories" : categories
        }
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=None)

class DebugDataset(Dataset):
    def __init(self, mode="bboxes"):
        self.mode = mode


    def __len__(self):
        return 25000
    
    def __getitem__(self, index):
        #random_tensor = torch.rand(1, 3, 224, 224)
        sample = {
               'image': torch.rand(3, 224, 224),
               'label': torch.tensor(2, dtype=torch.long)  # Return the category ID as label
            }
        return sample

class CustomCocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None, mode="bboxes"):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.mode = mode
        # Cache image info for faster access
        self.image_info_cache = {img["id"]: img for img in self.coco.loadImgs(self.image_ids)}

        self.loading_info = {}

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.image_info_cache[image_id]
        image_path = os.path.join(self.root_dir, image_info['gt_&_city'][1].replace("\\", ""), 
                                  image_info['gt_&_city'][2], image_info["file_name"] + "_leftImg8bit.png")
        #image_path = os.path.join(self.root_dir, image_info['gt_&_city'][1], 
        #                          image_info['gt_&_city'][2], image_info["file_name"] + "_leftImg8bit.png")
        image = cv2.imread(image_path)
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        
        if not annotations:
            index = random.randint(0, len(self.image_ids) - 1)
            return self.__getitem__(index)
        
        if self.mode == "bboxes":
            
            bboxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32)
            labels = torch.tensor([ann['category_id'] for ann in annotations], dtype=torch.long)

            bboxes[:, 2:] += bboxes[:, :2]

            image = Image.fromarray(image)
            
            target = {
                'boxes' : bboxes,
                'labels' : labels,
                'orig_size' : torch.tensor((2048, 1024)),
                'image_id' : torch.tensor(image_id)
            }

            if self.transforms:
                image, target = self.transforms(image, target)

            return image, target
        
        else:
            bboxes = torch.tensor([ann['bbox'] for ann in annotations])
            areas = bboxes[:, 2] * bboxes[:, 3]
            valid_indices = torch.where(areas >= 10_000)[0]

            if valid_indices.numel() > 0:
                idx = random.choice(valid_indices.tolist())
                ann = annotations[idx]
                bbox = ann['bbox']
                category_id = ann['category_id']
                x_min, y_min, width, height = bbox

            cropped_image_tensor = self.transforms(image)
            cropped_image_tensor = v2.functional.resized_crop(inpt=cropped_image_tensor, top=y_min, left=x_min, height=height, width=width, size=(224, 224), antialias=True),

            #Prepare the sample to return
            sample = {
               'image': cropped_image_tensor[0],
               'label': torch.tensor(category_id, dtype=torch.long)  # Return the category ID as label
            }


            return sample


