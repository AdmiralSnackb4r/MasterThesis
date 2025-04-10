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
import datasets.transforms as T
from collections import OrderedDict

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--datapath', default="/p/scratch/hai_1008/kromm3/CityScapes/leftImg8bit", help='path to data')

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
    parser.add_argument('--eval', default='/p/scratch/hai_1008/kromm3/RelTR/eval/run_4/eval.json', help='place to store evaluation')
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



def make_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    #T.RandomSizeCrop(384, 600), # TODO: cropping causes that some boxes are dropped then no tensor in the relation part! What should we do?
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=None),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)

def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to('cuda')
        return b

def compute_iou(box1, box2):
    xA = torch.max(box1[0], box2[0])
    yA = torch.max(box1[1], box2[1])
    xB = torch.min(box1[2], box2[2])
    yB = torch.min(box1[3], box2[3])

    inter_area = (xB - xA).clamp(0) * (yB - yA).clamp(0)

    box1_area = (box1[2] - box1[0]).clamp(0) * (box1[3] - box1[1]).clamp(0)
    box2_area = (box2[2] - box2[0]).clamp(0) * (box2[3] - box2[1]).clamp(0)

    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)

def match_predictions(preds, targets, iou_threshold=0.5):
    matched = set()
    tp = 0
    for pred_idx, pred_box in enumerate(preds['pred_boxes']):
        pred_label = preds['labels'][pred_idx]
        for tgt_idx, tgt_box in enumerate(targets['boxes']):
            if tgt_idx in matched:
                continue
            if pred_label != targets['labels'][tgt_idx]:
                continue
            iou = compute_iou(pred_box.tolist(), tgt_box.tolist())
            if iou > iou_threshold:
                tp += 1
                matched.add(tgt_idx)
                break
    return tp, len(preds['pred_boxes']), len(targets['boxes'])

def evaluate_model(predictions, ground_truths, confidence_threshold=0.95, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    iou_scores = []

    for pred, gt in zip(predictions, ground_truths):
        pred_logits = pred['pred_logits']  # [num_queries, num_classes]
        pred_boxes = pred['pred_boxes']    # [num_queries, 4] in cxcywh (normalized)

        probs = pred_logits.softmax(-1)[0, :, :-1]
        keep = torch.tensor(probs.max(-1).values > confidence_threshold)

        filtered_boxes = pred_boxes[0, keep]

        if filtered_boxes.numel() == 0:
            false_negatives += len(gt[0]['boxes'])
            continue

        # Convert predicted boxes to xyxy and scale to pixel space
        filtered_boxes = box_cxcywh_to_xyxy(filtered_boxes.to('cuda'))

        img_w, img_h = gt[0]['orig_size'].to('cuda')

        filtered_boxes = filtered_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to('cuda')

        #filtered_boxes = rescale_bboxes(filtered_boxes, orig_size)

        # Targets already in xyxy, absolute coordinates
        target_boxes = gt[0]['boxes'].to('cuda')
        target_labels = gt[0]['labels'].to('cuda')

        #print("pred", filtered_boxes, "target", target_boxes)

        matched_preds = set()
        matched_gts = set()

        #print(len(filtered_boxes), len(target_boxes))

        for pred_idx, pred_box in enumerate(filtered_boxes):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(target_boxes):
                if gt_idx in matched_gts:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and probs[keep][pred_idx].argmax() == target_labels[best_gt_idx]:
                true_positives += 1
                iou_scores.append(best_iou.item())
                matched_preds.add(pred_idx)
                matched_gts.add(best_gt_idx)
            else:
                false_positives += 1

        false_negatives += target_boxes.size(0) - len(matched_gts)
    
    print(true_positives, false_negatives, false_positives)

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    result = {
        'precision' : precision,
        'recall' : recall,
        'mean_iou' : mean_iou
    }

    return result


def evaluate(model, dataloader, device='cuda'):
    model.eval()
    total_fp = 0
    total_tp = 0
    total_fn = 0

    for samples, targets in dataloader:
        samples = samples.to(device)
        with torch.no_grad():
            outputs = model(samples)
        outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
        targets = [{k: v for k, v in t.items()} for t in targets]

        for pred, gt in zip (outputs, targets):
            tp, num_preds, num_gts = match_predictions(pred, gt)
            total_tp += tp
            total_fp += (num_preds - tp)
            total_fn += (num_gts - tp)
        break

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    result = {
        'precision' : precision,
        'recall' : recall,
        'f1' : f1
    }

    return result



def find_checkpoints(args):
    checkpoints = []
    for root, _, files in os.walk(args.resume):
        for file in files:
            #print(file)
            if file.endswith('.pth'):
                checkpoints.append(os.path.join(root, file))
    return checkpoints

def main(args):
    #utils.init_distributed_mode(args)
    device = torch.device(args.device)

    dataset_test = build_custom_dataset(args=args, anno_file='/p/project/hai_1008/kromm3/TrainOnCityScapes/CityScapes/annotations/train_dataset.json', transform=make_transforms('val'))
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_val = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    model, criterion, _ = custom_build_model(args)
    model.to(device)
    checkpoint = torch.load('/p/home/jusers/kromm3/jureca/master/scratch/RelTR/ckpt/run_4/checkpoint0114_.pth', map_location='cpu')
    new_state_dict = OrderedDict()
    state_dict = checkpoint['model']
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    # outputs_arr = []
    # targets_arr = []
    # model.eval()

    # with torch.no_grad():
    #     for samples, targets in data_loader_val:
    #         samples = samples.to(device)
    #         targets_arr.append(targets)  # still on CPU
    #         pred = model(samples)

    #         output_to_cpu = {
    #             'pred_logits' : pred['pred_logits'].cpu(),
    #             'pred_boxes' : pred['pred_boxes'].cpu()
    #         }
    #         outputs_arr.append(output_to_cpu)

    #         del samples
    #         del targets
    #         del pred
    #         torch.cuda.empty_cache()

    # del model
        
    # results = evaluate_model(outputs_arr, targets_arr)
    # print(results)


    results = evaluate(model, data_loader_val, 'cuda')
    print(results)

    os.makedirs(os.path.dirname(args.eval), exist_ok=True)


    with open(args.eval, 'w') as json_file:
        json.dump(results, json_file, indent=4)
        print(f"JSON saved to {args.eval}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RelTR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)  

