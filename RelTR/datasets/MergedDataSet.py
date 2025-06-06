import os
import re
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


class MergedCocoDataset(Dataset):
    def __init__(self, json_path, image_root, transforms=None):
        super().__init__()
        self.transforms = transforms

        # Load annotations
        with open(json_path, 'r') as f:
            coco = json.load(f)

        self.image_root = image_root
        self.images = coco['images']
        self.annotations = coco['annotations']
        self.categories = {cat['id']: cat['name'] for cat in coco['categories']}
        self.max_length = 0
        self.average_length = 0
        self.itter_counter = 0

        # Index annotations by image_id
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        # Optional: map file names to paths
        self.id_to_filename = {img['id']: img['file_name'] for img in self.images}

    def __len__(self):
        return len(self.images)
    
    def parse_filename(self, filename):

        if 'bdd' in filename:
            parts = filename.split("_", 2)
            if len(parts) < 3:
                raise ValueError(f"Unexpected filename format: {filename}")
            dataset = parts[0]
            typo = parts[1]
            img_name = parts[2]
            return dataset, typo, img_name, None
        elif 'city' in filename:
            folder, filename = os.path.split(filename)
    
            # folder is like "city_test_aachen"
            folder_parts = folder.split("_", 2)
            if len(folder_parts) != 3:
                raise ValueError(f"Unexpected folder structure: {folder}")
            
            dataset = folder_parts[0]        # "city"
            typo = folder_parts[1]          # "test"
            city = folder_parts[2]           # "aachen"
            
            # filename is like "aachen_000000_000019_leftImg8bit.png"
            img_name = "_".join(filename.split("_")[1:])  # "000000_000019_leftImg8bit.png"

            return dataset, typo, img_name, city
        
        elif 'mappilary' in filename:
            parts = filename.split('_', 2)
            if len(parts) < 3:
                raise ValueError(f"Unexpected folder structure: {folder}")
            dataset = parts[0]
            typo = parts[1]
            img_name = parts[2]

            return dataset, typo, img_name, None


        elif 'carla' in filename:
            parts = filename.split('_', 2)
            dataset = parts[0]
            typo = parts[1]
            img_name = parts[2]
            #print(parts)

            return dataset, typo, img_name, None
    
    def parse_path(self, dataset, typo, img_name, city):

        def strip_bdd_suffix(filename):
            name = re.sub(r'^(train|val|test)_', '', filename)
            # This regex removes `_train_color`, `_val_color`, `_test_color` (before the extension)
            name = re.sub(r'_(train|val|test)_color', '', name)
            # Replace .png with .jpg
            return os.path.splitext(name)[0] + '.jpg'

        if dataset == 'bdd':
            image_path = os.path.join(self.image_root, 'BDD100', 'bdd100k_images_10k', '10k', typo, strip_bdd_suffix(img_name))
        elif dataset == 'city':
            img_name = city + '_' + img_name
            typo = 'val' if 'test' in typo else 'train'
            image_path = os.path.join(self.image_root, 'CityScapes', 'leftImg8bit', typo, city, img_name)
        elif dataset == 'mappilary':
            typo = 'validation' if 'val' in typo else 'training'
            image_path = os.path.join(self.image_root, 'Mappilary', typo, 'images', img_name) 
        elif dataset == 'carla':
            filtered = 'filtered'
            if 'augmented' in img_name:
                top_lvl = '_aug'
                folder_nr, rest = img_name.split('_Aug', 1)
                folder = typo + '_' + folder_nr
                rest_parts = rest.lstrip('_').split('_')
                image_name_parts = []
                for part in rest_parts:
                    image_name_parts.append(part)
                    if '.png' in part:
                        break
                image_name = '_'.join(image_name_parts)

                image_path = os.path.join(self.image_root, 'Carla', 'Data', top_lvl, folder, filtered, image_name)
            else:
                top_lvl = '_out'
                parts = img_name.split('_')
                img_type = 'rgb'


                for i in reversed(range(len(parts))):
                    if parts[i].endswith('.png'):
                        png_part = parts[i]
                        break

                # The image ID is the part just before the last suffix (remove .png)
                image_id = parts[i - 1]
                image_name = f"{image_id}.png"

                # folder_nr is everything before the image ID
                folder_nr = '_'.join(parts[:i - 1])
                folder = typo + '_' + folder_nr

                if folder == "NoSignalJunctionCrossing_":
                    folder = "NoSignalJunctionCrossing"

                image_path = os.path.join(self.image_root, 'Carla', 'Data',  top_lvl, folder, img_type, filtered, image_name)
                #print(dataset, typo, img_name)

        return image_path

    def __getitem__(self, idx):
        self.itter_counter += 1
        image_info = self.images[idx]
        image_id = image_info['id']
        dataset, typo, img_name, city = self.parse_filename(image_info['file_name'])

        image = Image.open(self.parse_path(dataset, typo, img_name, city)).convert("RGB")
        annots = self.image_id_to_annotations.get(image_id, [])
        width, height = image.size

        boxes = []
        labels = []

        for ann in annots:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if len(boxes) == 0:
            index = random.randint(0, len(self.images) - 1)
            return self.__getitem__(index)


        if len(boxes) > self.max_length:
            print("new max length found: ", len(boxes))
            self.max_length = len(boxes)
        self.average_length += len(boxes)
        #print(f"length: {len(boxes)}")
        #print(f"average length: {self.average_length/self.itter_counter}")
        #print("len: ", len(boxes))
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

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import random

    dataset = MergedCocoDataset(
    json_path="annotations/Merged/merged_with_carla_train.json",
    image_root="/p/scratch/hai_1008/kromm3",  # Root folder to images
    transforms=None  # Or custom transform pipeline
    )

    for idx in range(len(dataset)):
        img, target = dataset[idx]
        print(idx)

    # # Load one sample
    # idx = random.randint(0, len(dataset) - 1)
    # img, target = dataset[idx]
    # print(img.size)  # PIL Image size
    # print(target)

    # # Draw bounding boxes
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.imshow(img)
    # #ax.set_title(f"Image: {target['file_name']}", fontsize=14)
    # for box, label in zip(target["boxes"], target["labels"]):
    #     x1, y1, x2, y2 = box
    #     width = x2 - x1
    #     height = y2 - y1

    #     rect = patches.Rectangle(
    #         (x1, y1), width, height,
    #         linewidth=2,
    #         edgecolor='red',
    #         facecolor='none'
    #     )
    #     ax.add_patch(rect)
    #     ax.text(
    #         x1, y1 - 5,
    #         str(label),
    #         color='white',
    #         fontsize=9,
    #         bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.2')
    #     )

    # ax.axis('off')
    # plt.tight_layout()
    # plt.show()