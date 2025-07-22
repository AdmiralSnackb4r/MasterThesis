import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class GTADataset(Dataset):

    def __init__(self, json_path, image_root, transforms=None):
        super().__init__()
        self.transforms = transforms

        with open(json_path, 'r') as f:
            coco = json.load(f)
        
        self.image_root = image_root
        self.images = coco['images']
        self.annotations = coco['annotations']
        self.categories = {cat['id']: cat['name'] for cat in coco['categories']}
        self.image_id_to_annotations = {}

        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_info = self.images[index]
        image_id = image_info['id']
        image = Image.open(os.path.join(self.image_root, image_info['file_name']))
        annots = self.image_id_to_annotations.get(image_id, [])
        width, height = image.size

        boxes = []
        labels = []

        for ann in annots:
            bbox = ann['bbox']
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id']+1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([image_id])

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'orig_size' : torch.tensor((width, height))
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target