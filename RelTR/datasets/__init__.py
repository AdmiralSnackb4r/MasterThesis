# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

#from .coco import build as build_coco, make_coco_transforms
from .CustomCocoDataset import CustomCocoDataset
from .CarlaDataset import CarlaDataset


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


#def build_dataset(image_set, args):
#    if args.dataset == 'vg' or args.dataset == 'oi':
#        return build_coco(image_set, args)
#    raise ValueError(f'dataset {args.dataset} not supported')


def build_custom_dataset(args, anno_file, transform=None):
    return CustomCocoDataset(root_dir=args.datapath, annotation_file=anno_file, mode='bboxes', transforms=transform)

def build_carla_dataset(args, anno_file, transform=None):
    return CarlaDataset(root_dir=args.datapath, annotation_file=anno_file, transforms=transform)