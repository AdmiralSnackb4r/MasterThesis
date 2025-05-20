import os
import re
import torch
import json
from torch.utils.data import Dataset
import cv2
from PIL import Image
import random
import glob

class Preprocess():
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.processed_dataset = {}

        self.class_id = {
                "ground" : 0,
                "road" : 1,
                "side walk" : 2,
                "bridge" : 3,
                "pole" : 4,
                "traffic light" : 5,
                "traffic sign" : 6,
                "person" : 7,
                "car" : 8,
                "truck" : 9,
                "bicycle" : 10
        }

        self.pred_id = {
                "on" : 0,
                "attached to" : 1,
                "on right side of" : 2,
                "parking on" : 3,
                "on left side of" : 4,
                "same road line as" : 5,
                "on right lane of" : 6,
                "on opposing side of" : 7,
                "on left lane of" : 8,
                "driving from right to left" : 9,
                "driving from left to right" : 10,
                "on middle lane of" : 11,
                "infront of" : 12,
                "behind" : 13,
                "riding" : 14,
                "next to" : 15,
                "turning right on" : 16,
                "driving on" : 17,
                "turning left on" : 18,
                "is hitting" : 19
        }

        self.load_annotation()

    def build_target(self, entry):
        entity_map = {}
        boxes = []
        labels = []

        rel_annotations = []

        for rel in entry['relationships']:
            for role in ['subject', 'object']:
                entity = rel[role]
                name = entity['name']
                bbox = entity['bbox']
                key = (name, tuple(bbox))

                if key not in entity_map:
                    x1, x2, y1, y2 = bbox
                    # cx = ((x1 + x2) / 2)
                    # cy = ((y1 + y2) / 2)
                    # w = (x2 - x1)
                    # h = (y2 - y1)

                    entity_map[key] = len(boxes)
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.class_id[name])

            sub_idx = entity_map[(rel['subject']['name'], tuple(rel['subject']['bbox']))]
            obj_idx = entity_map[(rel['object']['name'], tuple(rel['object']['bbox']))]
            rel_idx = self.pred_id[rel['predicate']]

            rel_annotations.append([sub_idx, obj_idx, rel_idx])

        return {
            "labels": labels,
            "boxes": boxes, 
            "rel_annotations": rel_annotations, 
            }
    

    def pre_process(self):


        for id in self.image_ids:
            entry = self.dataset[id]
            target = self.build_target(entry)
            self.processed_dataset[id] = target
        
        with open("processed_annotations.json", "w") as f:
            json.dump(self.processed_dataset, f)

    def load_annotation(self):
        with open(self.annotation_file, "r") as f:
            self.dataset = json.load(f)
        self.image_ids = list(self.dataset.keys())


class CarlaDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None, seg_ins_use=False):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.seg_ins_use = seg_ins_use
        self.load_annotation()
        self.transforms = transforms


    
    def load_annotation(self):
        with open(self.annotation_file, "r") as f:
            self.dataset = json.load(f)
        self.image_ids = list(self.dataset.keys())
        
    def __len__(self):
        return len(self.dataset)
    
    def load_random_version(self, root_dir, folder_name, base_name):
        pattern = os.path.join(root_dir, folder_name, f"{base_name}_*")
        candidates = glob.glob(pattern)
        
        if not candidates:
            raise FileNotFoundError(f"No versions found for {base_name} in {folder_name}")

        selected_path = random.choice(candidates)
        return selected_path
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]

        if not self.seg_ins_use:
            img_type = 'rgb'

        match = re.match(r"^(.*?)(0\d+)", image_id)
        folder_name = match.group(1).rstrip('_')  # remove trailing underscore if needed
        image_name = match.group(2)
        filtered = "filtered"

        if "augmented" in image_id:
            top_lvl = '_aug'
            folder_name = folder_name.split('_Aug')[0]
            image_name = 'augmented_' + image_name

            path = self.load_random_version(
                root_dir=os.path.join(self.root_dir, top_lvl),
                folder_name=os.path.join(folder_name, filtered),
                base_name=image_name
            )
            image = cv2.imread(path)
        else:
            top_lvl = '_out'
            image = cv2.imread(os.path.join(self.root_dir, top_lvl, folder_name, img_type, filtered, image_name + ".png"))
        
        target = self.dataset[image_id]
        #print(match)
        image_id = torch.tensor([index])

        target = {
            "labels" : torch.tensor(target['labels'], dtype=torch.long),
            "boxes" : torch.tensor(target['boxes'], dtype=torch.float32),
            "rel_annotations" : torch.tensor(target['rel_annotations'], dtype=torch.long),
            "image_id" : image_id,
            "orig_size" : torch.tensor((1920, 1080))
        }

        #print(target)

        image = Image.fromarray(image)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
    


if __name__ == '__main__':

    preprocesser = Preprocess(root_dir="F:\\scenario_runner-0.9.15", annotation_file='datasets\\annotations\\Carla\\train_dataset.json')
    preprocesser.pre_process()