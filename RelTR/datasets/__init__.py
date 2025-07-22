# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

#from .coco import build as build_coco, make_coco_transforms
from .CustomCocoDataset import CustomCocoDataset
from .CarlaDataset import CarlaDataset
from .MergedDataSet import MergedCocoDataset
from .GTA_Dataset import GTADataset

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

def build_carla_dataset(args, anno_carla, anno_real = None, transform=None, pseudo=False):
    return CarlaDataset(root_dir_carla=args.datapath_carla, anno_carla=anno_carla, anno_real=anno_real, root_dir_real=args.datapath_real, transforms=transform, pseudo=pseudo)

def build_merged_dataset(args, anno_file, transform=None):
    return MergedCocoDataset(image_root=args.datapath, json_path=anno_file, transforms=transform)

def build_gta_dataset(args, anno_file, transform=None):
    return GTADataset(json_path=anno_file, image_root=args.datapath, transforms=transform)