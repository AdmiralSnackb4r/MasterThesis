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
    def __init__(self, root_dir_carla, anno_carla, anno_real=None, root_dir_real=None,
                transforms=None, seg_ins_use=False, pseudo=False):
        self.root_dir_carla = root_dir_carla
        self.anno_carla = anno_carla
        self.anno_real = anno_real
        self.seg_ins_use = seg_ins_use
        self.root_dir_real = root_dir_real
        self.pseudo = pseudo
        self.load_annotation()
        self.transforms = transforms
        self.use_real_now = False
        # Categories for evaluation
        self.rel_categories = ['N/A', 'on', 'attached to', 'on right side of', 'parking on', 'on left side of', 'same road line as', 'on right lane of',
                            'on opposing side of', 'on left lane of', 'driving from right to left', 'driving from left to right', 'on middle lane of',
                            'infront of', 'behind', 'riding', 'next to', 'turning right on', 'driving on', 'turning left on', 'is hitting']


    
    def load_annotation(self):
        with open(self.anno_carla, "r") as f:
            self.dataset_carla = json.load(f)
        self.image_ids_carla = list(self.dataset_carla.keys())

        if self.anno_real:
            with open(self.anno_real, "r") as f:
                self.dataset_real = json.load(f)
            if not self.pseudo:
                self.real_images = self.dataset_real['images']
                self.annotations_real = self.dataset_real['annotations']

                # Index annotations by image_id
                self.image_id_to_annotations = {}
                for ann in self.annotations_real:
                    img_id = ann['image_id']
                    if img_id not in self.image_id_to_annotations:
                        self.image_id_to_annotations[img_id] = []
                    self.image_id_to_annotations[img_id].append(ann)

                # Optional: map file names to paths
                self.id_to_filename = {img['id']: img['file_name'] for img in self.real_images}
            else:
                self.real_images = list(self.dataset_real.keys())
        else:
            self.dataset_real = None
            self.image_ids_real = []
        
    def __len__(self):
        if not self.dataset_real:
            return len(self.image_ids_carla)
        # Ensure even interleaving
        return 2 * max(len(self.image_ids_carla), len(self.real_images))
    
    def load_real_image(self, image_id):
        dataset, typo, img_name, city = self.parse_filename(image_id)
        image = Image.open(self.parse_path(dataset, typo, img_name, city)).convert("RGB")
        return image
    
    def load_image(self, image_id):
        if not self.seg_ins_use:
            img_type = 'rgb'

        match = re.match(r"^(.*?)(0\d+)", image_id)
        folder_name = match.group(1).rstrip('_')
        image_name = match.group(2)
        filtered = "filtered"

        if "augmented" in image_id:
            top_lvl = '_aug'
            folder_name = folder_name.split('_Aug')[0]
            image_name = 'augmented_' + image_name
            path = self.load_random_version(
                root_dir=os.path.join(self.root_dir_carla, top_lvl),
                folder_name=os.path.join(folder_name, filtered),
                base_name=image_name
            )
        else:
            top_lvl = '_out'
            path = os.path.join(self.root_dir_carla, top_lvl, folder_name, img_type, filtered, image_name + ".png")

        image = cv2.imread(path)
        return Image.fromarray(image).convert("RGB")
    
    def load_random_version(self, root_dir, folder_name, base_name):
        pattern = os.path.join(root_dir, folder_name, f"{base_name}_*")
        candidates = glob.glob(pattern)
        
        if not candidates:
            raise FileNotFoundError(f"No versions found for {base_name} in {folder_name}")

        selected_path = random.choice(candidates)
        return selected_path
    
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
            image_path = os.path.join(self.root_dir_real, 'BDD100', 'bdd100k_images_10k', '10k', typo, strip_bdd_suffix(img_name))
        elif dataset == 'city':
            #img_name = city + '_' + img_name
            if self.pseudo:
                image_path = os.path.join(self.root_dir_real, 'CityScapes', 'leftImg8bit', typo, city, img_name+"_leftImg8bit.png")
            else:
                img_name = city + '_' + img_name
                typo = 'val' if 'test' in typo else 'train'
                image_path = os.path.join(self.root_dir_real, 'CityScapes', 'leftImg8bit', typo, city, img_name+".png")
        elif dataset == 'mappilary':
            typo = 'validation' if 'val' in typo else 'training'
            image_path = os.path.join(self.root_dir_real, 'Mappilary', typo, 'images', img_name) 
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

                image_path = os.path.join(self.root_dir_real, top_lvl, folder, filtered, image_name)
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

                image_path = os.path.join(self.image_root, top_lvl, folder, img_type, filtered, image_name)
                #print(dataset, typo, img_name)

        return image_path
    
    def __getitem__(self, index):
        use_real = self.dataset_real and self.use_real_now
        #use_real = True
        if use_real:
            real_idx = (index // 2) % len(self.real_images)
            image_info = self.real_images[real_idx]


            if not self.pseudo:
                image_id = image_info['id']
                dataset, typo, image_name, city = self.parse_filename(image_info['file_name'])

                image = Image.open(self.parse_path(dataset, typo, image_name, city)).convert("RGB")
                annots = self.image_id_to_annotations.get(image_id, [])

                boxes = []
                labels = []

                for ann in annots:
                    bbox = ann['bbox']  # COCO format: [x, y, width, height]
                    x, y, w, h = bbox
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'])

                if len(boxes) == 0:
                    index = random.randint(0, len(self.real_images) - 1)
                    return self.__getitem__(index)

                target = {
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "rel_annotations":  torch.full((1, 3), -1, dtype=torch.long), #placeholder
                    "image_id": torch.tensor([index]),
                    "orig_size": torch.tensor((2048, 1024)),
                }
            else:
                annotations = self.dataset_real[image_info]
                image = Image.open(self.parse_path('city', typo=annotations['gt_&_city'][1], img_name=image_info, city=annotations['gt_&_city'][2]))
                image_id = image_info

                labels = annotations['labels']
                boxes = annotations['boxes']
                rel_annotations = annotations['rel_annotations']

                target = {
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "rel_annotations":  torch.tensor(rel_annotations, dtype=torch.long),
                    "image_id": torch.tensor([index]),
                    "orig_size": torch.tensor((2048, 1024)),
                }

        else:
            carla_idx = index #(index // 2) % len(self.image_ids_carla)
            image_id = self.image_ids_carla[carla_idx]
            target_data = self.dataset_carla[image_id]

            fixed_labels = [label + 1 for label in target_data['labels']]
            fixed_rel_annotations = [[sub, obj, rel + 1] for sub, obj, rel in target_data['rel_annotations']]

            image = self.load_real_image(image_id)
            target = {
                "labels": torch.tensor(fixed_labels, dtype=torch.long),
                "boxes": torch.tensor(target_data['boxes'], dtype=torch.float32),
                "rel_annotations": torch.tensor(fixed_rel_annotations, dtype=torch.long),
                "image_id": torch.tensor([index]),
                "orig_size": torch.tensor((1920, 1080)),

                }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target, image_id
    


if __name__ == '__main__':

    preprocesser = Preprocess(root_dir="F:\\scenario_runner-0.9.15", annotation_file='datasets\\annotations\\Carla\\train_dataset.json')
    preprocesser.pre_process()