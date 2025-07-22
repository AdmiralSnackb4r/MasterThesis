import argparse
import torch
import datasets.transforms as T
import util.misc as utils
from models import custom_build_model, build_model
from datasets import build_custom_dataset, build_merged_dataset, build_carla_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
import json
import math

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    #parser.add_argument('--datapath', default="/p/scratch/hai_1008/kromm3/CityScapes/leftImg8bit", help='path to data')
    parser.add_argument('--datapath', default="S:\\Datasets", help='path to data')
    parser.add_argument('--datapath_carla', default='F:\\scenario_runner-0.9.15\\Data', type=str)
    parser.add_argument('--datapath_real', default='S:\\Datasets', type=str)

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
    parser.add_argument('--num_entities', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt\\run_pseudo\\checkpoint_re_0359.pth', help='resume from checkpoint')
    #parser.add_argument('--eval', default='/p/scratch/hai_1008/kromm3/RelTR/eval/run_4/eval.json', help='place to store evaluation')
    parser.add_argument('--eval', default='RelTR\\eval\\run_12\\eval.json', help='place to store evaluation')
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

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)

def make_transforms(image_set):

    normalize = T.Compose([
        T.RandomResize([800], max_size=1333),
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
            #T.RandomResize([800], max_size=None),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def main(args):

    device = torch.device(args.device)

    dataset_test = build_carla_dataset(args=args,
                                        anno_carla='datasets\\annotations\\CityScapes\\converted_relationship_dataset.json', transform=make_transforms('val'))

    ckpt = torch.load(args.resume, weights_only=False)
    sampler_test = SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    model, _, _ = build_model(args)
    model.to(device)

    new_state_dict = OrderedDict()
    state_dict = ckpt['model']
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)

    model.eval()
    evaluator_list = []

    evaluation_out = {
        # "all_classes" : {
        #     'R@20' : [],
        #     'R@50' : [],
        #     'R@100' : [],
        # },
    }

    for index, name in enumerate(data_loader_test.dataset.rel_categories):
        evaluator_list.append((index, name, BasicSceneGraphEvaluator(mode='sgdet')))
        evaluation_out[name] =  {
            'R@20' : [],
            'R@50' : [],
            'R@100' : [],
        }

    evaluator = BasicSceneGraphEvaluator(mode='sgdet')
    counter = 0

    #print(evaluation_out)
    def iou(box1, box2):
        # box = [xmin, ymin, xmax, ymax]
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0


    def box_center(box):
        """Calculate center (cx, cy) of a bounding box."""
        x_min, y_min, x_max, y_max = box
        return [(x_min + x_max) / 2, (y_min + y_max) / 2]

    def euclidean_dist(c1, c2):
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def merge_predictions(net1_boxes, net2_boxes, net1_labels, net2_labels):
        result = []

        # print(net1_labels, net1_labels.shape)

        # print("-----------")

        # print(net2_labels, net2_labels.shape)

        for i, box2 in enumerate(net2_boxes):
            label2 = net2_labels[i].item()
            box2_center = box_center(box2)

            # Filter net1 boxes by same label
            candidate_indices = [j for j, lbl in enumerate(net1_labels) if lbl.item() == label2]

            # Further filter only boxes that overlap (iou > 0) with box2
            overlapping_indices = []
            for j in candidate_indices:
                box1 = net1_boxes[j]
                if iou(box1, box2) > 0:
                    overlapping_indices.append(j)

            # If none overlap, fallback: use all candidates with same label
            #if not overlapping_indices:
            #    overlapping_indices = candidate_indices

            # If still no candidates, keep box2 as is
            if not overlapping_indices:
                result.append(box2)
                continue

            # Find the closest one among overlapping candidates
            min_dist = float('inf')
            best_idx = -1
            for j in overlapping_indices:
                box1 = net1_boxes[j]
                dist = euclidean_dist(box_center(box1), box2_center)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = j

            # Merge with closest one
            merged = net1_boxes[best_idx]
            result.append(merged)

        return torch.stack(result)


    with torch.no_grad():
        for samples, targets, image_ids in tqdm(data_loader_test, desc="Collecting Ouputs", unit="batch"):
            counter += 1
            targets = targets[0]
            samples.to(device)

            gt_entry = {
                'gt_relations' : targets['rel_annotations'].cpu().numpy(),
                'gt_boxes' : box_cxcywh_to_xyxy(targets['boxes']).cpu().numpy(),
                'gt_classes' : targets['labels'].cpu().numpy()
            }
            outputs = model(samples)

            #print(outputs['sub_logits'].shape)

            pred_logits = outputs['pred_logits'].softmax(-1)[0, :, :-1].argmax(dim=1)
            pred_boxes = outputs['pred_boxes'].squeeze(0)

            sub_bboxes = box_cxcywh_to_xyxy(outputs['sub_boxes']).cpu().clone()
            obj_bboxes = box_cxcywh_to_xyxy(outputs['obj_boxes']).cpu().clone()

            sub_bboxes = sub_bboxes.permute(0, 2, 1).reshape(-1, 4)  # (1, 200, 4) → (200, 4)
            obj_bboxes = obj_bboxes.permute(0, 2, 1).reshape(-1, 4)  # (1, 200, 4) → (200, 4)



            pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'].softmax(-1)[:, :, :-1], dim=2)
            pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'].softmax(-1)[:, :, :-1], dim=2)

            # sub_bboxes = merge_predictions(
            #     pred_boxes, sub_bboxes, pred_logits, pred_sub_classes.squeeze(0))#

            # obj_bboxes = merge_predictions(
            #     pred_boxes, obj_bboxes, pred_logits, pred_obj_classes.squeeze(0))

            rel_scores = outputs['rel_logits'][0][:, 1: -1].softmax(-1)

            pred_entry = {
                'sub_boxes' : sub_bboxes,
                'sub_classes' : pred_sub_classes.view(-1, 1).cpu().clone().numpy(),
                'sub_scores' : pred_sub_scores.view(-1, 1).cpu().clone().numpy(),
                'obj_boxes' : obj_bboxes,
                'obj_classes' : pred_obj_classes.view(-1, 1).cpu().clone().numpy(),
                'obj_scores' : pred_obj_scores.view(-1, 1).cpu().clone().numpy(),
                'rel_scores' : rel_scores.cpu().clone().numpy()
            }

            #print( targets['rel_annotations'].shape)

            #print("gt boxes: ", box_cxcywh_to_xyxy(targets['boxes']).cpu().numpy())
            #print("pred boxes: ", sub_bboxes)

            #res, output = evaluator.evaluate_scene_graph_entry(gt_entry, pred_entry)

            #print(image_ids[0])



            # evaluation_out[image_ids[0]] = {
            #                 'R@20' : output['R@20'],
            #                 'R@50' : output['R@50'],
            #                 'R@100' : output['R@100']
            #             }

            # evaluation_out["all_classes"]['R@20'].append(output['R@20'])
            # evaluation_out["all_classes"]['R@50'].append(output['R@50'])
            # evaluation_out["all_classes"]['R@100'].append(output['R@100'])

            if evaluator_list is not None:
                for pred_id, pred_name, evaluator_rel in evaluator_list:
                    #if pred_id == 17:
                        gt_entry_rel = gt_entry.copy()
                        mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                        gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                        if gt_entry_rel['gt_relations'].shape[0] == 0:
                            continue
                        #gt_entry_rel['gt_relations'][:, -1] += 1
                        #print("Relationship: ", pred_name)
                        #print("ground truth: ", gt_entry_rel)
                        #print("prediction: ", pred_entry)
                        _, output = evaluator_rel.evaluate_scene_graph_entry(gt_entry_rel, pred_entry)
                        #print("R@20: ", output['R@20'])
                        #print("R@50: ", output['R@50'])
                        #print("R@100: ", output['R@100'])
                        # evaluation_out[image_ids] = {
                        #     'R@20' : output['R@20'],
                        #     'R@50' : output['R@50'],
                        #     'R@100' : output['R@100']
                        # }
                        evaluation_out[pred_name]['R@20'].append(output['R@20'])
                        evaluation_out[pred_name]['R@50'].append(output['R@50'])
                        evaluation_out[pred_name]['R@100'].append(output['R@100'])
            #print(res)

            del samples
            torch.cuda.empty_cache()
        
        for entry in evaluation_out:
            evaluation_out[entry]['R@20'] = float(np.mean(evaluation_out[entry]['R@20'])) 
            evaluation_out[entry]['R@50'] = float(np.mean(evaluation_out[entry]['R@50'])) 
            evaluation_out[entry]['R@100'] = float(np.mean(evaluation_out[entry]['R@100']))

        with open("eval_pseudo_phrase_real.json", "w") as f:
            json.dump(evaluation_out, f, indent=4)
    







if __name__ == "__main__":
    parser = argparse.ArgumentParser('RelTR evaluation script for relationships', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)  