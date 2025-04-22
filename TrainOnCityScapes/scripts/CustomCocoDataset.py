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


    def __init__(self, root_dir, annotation_file, exclude_category_ids=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.dataset_size = len(self.image_ids)
        self.exclude_category_ids = set(exclude_category_ids or [])

        # self.image_ids = self._filter_images()
        # self.dataset_size = len(self.image_ids)

    # def _filter_images(self):
    #     valid_image_ids = []
    #     for img_id in self.coco.imgs:
    #         ann_ids = self.coco.getAnnIds(imgIds=img_id)
    #         anns = self.coco.loadAnns(ann_ids)

    #         if all(ann['category_id'] not in self.exclude_category_ids for ann in anns):
    #             valid_image_ids.append(img_id)
    #     return valid_image_ids

    def split_train_val_test(self):
        
        train_ids, val_ids, test_ids = [], [], []

        for img_id in self.image_ids:
            filename = self.coco.imgs[img_id]['file_name'].lower()
            if "weimar" in filename:
                val_ids.append(img_id)
            elif "zurich" in filename:
                test_ids.append(img_id)
            else:
                train_ids.append(img_id)

        return train_ids, val_ids, test_ids
    
    def create_split_annotations(self, split_ids, output_file):

        label_id = {
        "road" : 0,
        "side walk" : 1,
        "parking" : 2,
        "bridge" : 3,
        "pole" : 4,
        "traffic light" : 5,
        "traffic sign" : 6,
        "rider" : 7,
        "person" : 7,
        "car" : 8,
        "truck" : 9,
        "bus" : 10,
        "train" : 11,
        "motorcycle" : 12,
        "bicycle" : 13,
        "ground" : 14
        }


         # Invert label_id mapping for name lookup
        name_to_new_id = label_id
        new_id_to_name = {v: k for k, v in label_id.items()}

        # Get annotations and filter out excluded categories
        annotations = [
            ann for ann in self.coco.anns.values()
            if ann['image_id'] in split_ids and ann['category_id'] not in self.exclude_category_ids
        ]

        # Remap category IDs in annotations
        for ann in annotations:
            original_cat_id = ann['category_id']
            category_name = self.coco.cats[original_cat_id]['name']
            ann['category_id'] = name_to_new_id[category_name]

        # Collect only used category names
        used_category_names = set(self.coco.cats[ann['category_id']]['name'] for ann in annotations)

        # Build new categories list with updated IDs
        categories = [
            {"id": new_id, "name": name}
            for name, new_id in label_id.items()
            if name in used_category_names
        ]

        images = [self.coco.imgs[img_id] for img_id in split_ids]

        split_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories
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


