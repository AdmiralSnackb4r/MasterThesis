import cv2
from labels import labels, label_id
import numpy as np
from sample_annotation import coco_annotations
import os

path_to_dataset = 'S:\\Datasets\\CityScapes'
gtCoarse = 'gtCoarse'
gtFine = 'gtFine'
leftImg8bit = '\\leftImg8bit'

train = '\\train'
train_extra = '\\train_extra'
val = '\\val'

ground_truths = {
    gtFine : [train, val],
    gtCoarse : [train, train_extra, val]
}


def generate_bounding_boxes(class_path, instance_path):
    """
    Generates bounding boxes for each instance in the instance image with corresponding class IDs.
    
    Args:
        class_path (String): Path to class image
        instance_path (String): Path to instance image
        
    Returns:
        list: A list of dictionaries containing bounding boxes with class IDs. 
              Each dictionary has the format {'instance_id': int, 'class_id': int, 'bbox': (x, y, w, h)}.
    """
    class_image = cv2.imread(class_path, cv2.IMREAD_UNCHANGED)
    instance_image = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)

    instance_ids = np.unique(instance_image)
    bounding_boxes = []
    
    for instance_id in instance_ids:
        # Create a mask for the current instance
        mask = (instance_image == instance_id).astype(np.uint8)
        
        # Find contours in the instance mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            
            x, y, w, h = cv2.boundingRect(contour)
            # Get the class ID from the class_image using a pixel from the instance mask
            class_id = class_image[contour[0][0][1], contour[0][0][0]]  # Take the top-left corner pixel for class ID
            class_id = class_id[:-1][::-1]
            class_id = labels[tuple(class_id)]
            if class_id != "void":

            
                bounding_boxes.append({
                    'instance_id': int(instance_id),
                    'class_id': class_id,
                    'bbox': (x, y, w, h)
                })
    
    return bounding_boxes


def is_categorie_in_annotations(categorie, annotations):
    try:
        existing_categories = {categorie["name"] for categorie in annotations["categories"]}
    except KeyError: # First try needs to be catched
        return False
    return categorie in existing_categories

def is_image_in_annotations(image_name, annotations):
    try:
        existing_names = {image["file_name"] for image in annotations["images"]}
    except KeyError: # First try needs to be catched
        return False
    return image_name in existing_names


def annotate():
    id_image_counter = 0
    id_annotation_counter = 0


    for gt in ground_truths.keys():
        for folder in ground_truths[gt]:
            path = path_to_dataset + "\\"+ gt + folder
            city_folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
            for city in city_folders:
                files = [name for name in os.listdir(path+"\\"+city) if os.path.isfile(os.path.join(path+"\\"+city, name))]
                for file_name in files:
                    cutted_name = '_'.join(file_name.split('_')[:-2])
                    if not is_image_in_annotations(cutted_name, annotations=coco_annotations):
                        print(os.path.join(path+"\\"+city+"\\"+file_name))
                        image = cv2.imread(os.path.join(path+"\\"+city+"\\"+file_name))
                        annotations = generate_bounding_boxes(class_path=path+"\\"+city+"\\"+cutted_name+"_"+gt+"_color.png", 
                                                            instance_path=path+"\\"+city+"\\"+cutted_name+"_"+gt+"_instanceIds.png")
                        height, width, _ = image.shape
                        coco_annotations["images"].append({
                            "id" : id_image_counter,
                            "width" : width,
                            "height" : height,
                            "file_name" : cutted_name,
                            "gt_&_city" : [gt, folder, city],
                            "license" : "NONE",
                            "date_captured" : "NONE" 
                        })
                        boxes = [box['bbox'] for box in annotations]
                        class_ids = [box['class_id'] for box in annotations]
                        for (x, y, w, h), class_id in zip(boxes, class_ids):
                            coco_annotations["annotations"].append({
                                "id" : id_annotation_counter,
                                "image_id" : id_image_counter,
                                "category_id" : label_id[class_id],
                                "area" : w*h,
                                "bbox" : [x, y, w, h],
                                "iscrowd" : 0
                            })
                            id_annotation_counter += 1
                            if not is_categorie_in_annotations(class_id, annotations=coco_annotations):
                                coco_annotations["categories"].append({
                                    "id" : label_id[class_id],
                                    "name" : class_id
                                })
                    id_image_counter += 1
    return coco_annotations

if __name__ == "__main__":
    import json
    annos = annotate()
    with open("coco_annotations.json", "w") as f:
        json.dump(annos, f, indent=None)
