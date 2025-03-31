import os
import torch
import json
from pathlib import Path
import torchvision.transforms as T
import argparse
from models import custom_build_model
from datasets import build_custom_dataset
import util.misc as utils
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt/', help='resume from checkpoint')
    parser.add_argument('--eval', default='eval/eval_.json', help='place to store evaluation')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser

def compute_iou(box1, box2):
     # Calculate intersection coordinates
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # Compute IoU
    return inter_area / union_area if union_area > 0 else 0

def evaluate_model(predictions, ground_truths, iou_threshold=0.5, confidence_threshold=0.0):
    """
    Evaluates the model using IoU and mAP metrics.
    
    """
    pred_probs = torch.softmax(predictions['pred_logits'], dim=-1)
    pred_classes = pred_probs.argmax(dim=-1)
    pred_confidences = pred_probs.max(dim=-1).values
    
    mask = pred_confidences > confidence_threshold
    filtered_boxes = predictions['pred_boxes'][mask]
    filtered_classes = pred_classes[mask]
    filtered_confidences = pred_confidences[mask]

    true_positives, false_positives, false_negatives = 0, 0, 0
    iou_scores = []

    for target, pred_box, pred_class, pred_conf in zip(ground_truths, filtered_boxes, filtered_classes, filtered_confidences):
        target_boxes = target['boxes']
        target_labels = target['labels']
        pred_box = pred_box.unsqueeze(0)

        print(pred_box.shape, target_boxes.shape)

        ious = box_iou(pred_box, target_boxes)
        max_iou, max_iou_idx = ious.max(dim=1)
        print(max_iou, max_iou_idx)

        if max_iou > iou_threshold and pred_class == target_labels[max_iou_idx]:
            true_positives += 1
            iou_scores.append(max_iou.item())
        else:
            false_positives += 1
        
        false_negatives += len(target_boxes) - len(filtered_boxes)

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    mean_iou = np.mean(iou_scores) if iou_scores else 0.0

    return {
        'precision' : precision,
        'recall' : recall,
        'mean_iou' : mean_iou
    }



def find_checkpoints(args):
    checkpoints = []
    for root, _, files in os.walk(args.resume):
        for file in files:
            #print(file)
            if file.endswith('.pth'):
                checkpoints.append(os.path.join(root, file))
    return checkpoints

def load_model(checkpoint, model):
    ckpt = torch.load(checkpoint)
    #print(ckpt.keys())
    new_state_dict = {}
    for key, value in ckpt['model'].items():
        if "module" in key:
            new_key = key.replace("module.", "", 1)  # Remove 'module.' only from the beginning
        else:
            new_key = key

        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model

def main(args):
    #utils.init_distributed_mode(args)
    device = torch.device(args.device)
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
    ])

    dataset_val = build_custom_dataset("/mnt/s/Datasets/CityScapes/leftImg8bit", "./datasets/test_dataset.json", image_set='val')
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    model, criterion, _ = custom_build_model(args)

    checkpoints = find_checkpoints(args)
    #print(checkpoints)
    result_dir = {}

    for ckpt in checkpoints:
        model = load_model(ckpt, model)
        model.to(device)
        model.eval()
        result_dir[ckpt] = []

        for samples, targets in data_loader_val:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            results = evaluate_model(outputs, targets)
            result_dir[ckpt].append(results)
            break


    with open(args.eval, 'w') as json_file:
        json.dump(result_dir, json_file, indent=4)
        print(f"JSON saved to {args.eval}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RelTR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)  

