import json

coco_annotations = {
    "info": {
        "description": "Cityscapes to COCO Bounding Boxes",
        "version": "1.0",
        "year": 2024,
        "contributor": "Edward Kromm",
        "date_created": "2024-11-04"
    },
    "licenses": [
        {
            "id": 1,
            "name": "License Name",
            "url": "http://example.com/license"
        }
    ],
    "images": [
        #{
           # "id": 1,
           #"width": 640,
           #"height": 480,
           # "file_name": "image1.jpg",
           # "license": 1,
           # "date_captured": "2024-10-17"
        #}
    ],
    "annotations": [
        #{
           # "id": 1,
           # "image_id": 1,
           # "category_id": 1,
           # "area": 2500,
           # "bbox": [100, 100, 50, 50],
           # "iscrowd": 0
        #}
    ],
    "categories": [
        #{
           # "id": 1,
           # "name": "category1",
           # "supercategory": "supercategory1"
        #}
    ]
}