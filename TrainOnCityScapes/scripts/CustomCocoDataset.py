import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import os
import numpy as np
import random
import json


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


class CustomCocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None, mode="bboxes"):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['gt_&_city'][1].replace("\\", ""), 
                                  image_info['gt_&_city'][2], image_info["file_name"] + "_leftImg8bit.png")
        image = cv2.imread(image_path)
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        
        if not annotations:
            index = random.randint(0, len(self.image_ids) - 1)
            return self.__getitem__(index)
        
        if self.mode == "bboxes":
            boxes = []
            labels = []
            for ann in annotations:
                boxes.append(ann["bbox"])
                labels.append(ann["category_id"])

            # Convert bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max]
            boxes = np.array(boxes)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x_min + width
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y_min + height

            sample = {
                'image': torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0,  # Normalize the image
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long)
            }

            if self.transforms:
                sample = self.transforms(sample)

            return sample
        else:
            ann = random.choice(annotations)
            bbox = ann['bbox']
            category_id = ann["category_id"]
            # Convert bbox from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            # Crop the image using the bounding box
            cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Handle cases where the cropping might yield an empty image
            if cropped_image.size == 0:
                raise ValueError("Cropped image is empty. Please check bounding box coordinates.")
             # Convert to a tensor and normalize the image
            if self.transforms:
                cropped_image_tensor = self.transforms(cropped_image)

            # Prepare the sample to return
            sample = {
                'image': cropped_image_tensor,
                'label': torch.tensor(category_id, dtype=torch.long)  # Return the category ID as label
            }

            return sample


