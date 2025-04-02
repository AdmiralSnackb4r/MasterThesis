# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import numpy as np
import time
import torch

#from datasets.coco_eval import CocoEvaluator
import util.misc as utils
from util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from lib.openimages_evaluation import task_evaluation_sg

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    
    model.train()
    criterion.train()
    
    header = f'Epoch: [{epoch}]'
    print_freq = 500
    
    stats = {
        'loss': [],
        'class_error': [],
        'lr': []
    }
    
    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Reduce losses over all GPUs (if using distributed training)
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Collect stats
        stats['loss'].append(loss_value)
        stats['class_error'].append(loss_dict_reduced.get('class_error', 0))
        stats['lr'].append(optimizer.param_groups[0]['lr'])

        if batch_idx % print_freq == 0:
            avg_loss = sum(stats['loss']) / len(stats['loss'])
            avg_class_error = sum(stats['class_error']) / len(stats['class_error'])
            print(f"{header} [{batch_idx}/{len(data_loader)}]  Loss: {avg_loss:.4f}  Class Error: {avg_class_error:.2f}% ")

    return stats


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):

    model.eval()
    criterion.eval()

    header = f'Evaluate'
    print_freq = 500
    
    stats = {
        'loss': [],
        'class_error': []
    }

    for batch_idx, (samples, targets) in enumerate(data_loader):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = losses.item()

        # Collect stats
        stats['loss'].append(loss_value)
        stats['class_error'].append(loss_dict_reduced.get('class_error', 0))

        if batch_idx % print_freq == 0:
            avg_loss = sum(stats['loss']) / len(stats['loss'])
            avg_class_error = sum(stats['class_error']) / len(stats['class_error'])
            print(f"{header} [{batch_idx}/{len(data_loader)}]  Loss: {avg_loss:.4f}  Class Error: {avg_class_error:.2f}% ")

    return stats

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):
    pass
    # for batch, target in enumerate(targets):
    #     target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

    #     gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
    #                 #'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
    #                 'gt_boxes': target_bboxes_scaled}

    #     # sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
    #     # obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

    #     # pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
    #     # pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)
    #     # rel_scores = outputs['rel_logits'][batch][:,1:-1].softmax(-1)

    #     # pred_entry = {'sub_boxes': sub_bboxes_scaled,
    #     #               'sub_classes': pred_sub_classes.cpu().clone().numpy(),
    #     #               'sub_scores': pred_sub_scores.cpu().clone().numpy(),
    #     #               'obj_boxes': obj_bboxes_scaled,
    #     #               'obj_classes': pred_obj_classes.cpu().clone().numpy(),
    #     #               'obj_scores': pred_obj_scores.cpu().clone().numpy(),
    #     #               'rel_scores': rel_scores.cpu().clone().numpy()}

    #     # evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)

    #     if evaluator_list is not None:
    #         for pred_id, _, evaluator_rel in evaluator_list:
    #             gt_entry_rel = gt_entry.copy()
    #             mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
    #             gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
    #             if gt_entry_rel['gt_relations'].shape[0] == 0:
    #                 continue
    #             evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)


def evaluate_rel_batch_oi(outputs, targets, all_results):
    pass

    # for batch, target in enumerate(targets):
    #     target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

    #     sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
    #     obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

    #     pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
    #     pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

    #     rel_scores = outputs['rel_logits'][batch][:, :-1].softmax(-1)

    #     relation_idx = target['rel_annotations'].cpu().numpy()
    #     gt_sub_boxes = target_bboxes_scaled[relation_idx[:, 0]]
    #     gt_sub_labels = target['labels'][relation_idx[:, 0]].cpu().clone().numpy()
    #     gt_obj_boxes = target_bboxes_scaled[relation_idx[:, 1]]
    #     gt_obj_labels = target['labels'][relation_idx[:, 1]].cpu().clone().numpy()

    #     img_result_dict = {'sbj_boxes': sub_bboxes_scaled,
    #                        'sbj_labels': pred_sub_classes.cpu().clone().numpy(),
    #                        'sbj_scores': pred_sub_scores.cpu().clone().numpy(),
    #                        'obj_boxes': obj_bboxes_scaled,
    #                        'obj_labels': pred_obj_classes.cpu().clone().numpy(),
    #                        'obj_scores': pred_obj_scores.cpu().clone().numpy(),
    #                        'prd_scores': rel_scores.cpu().clone().numpy(),
    #                        'image': str(target['image_id'].item())+'.jpg',
    #                        'gt_sbj_boxes': gt_sub_boxes,
    #                        'gt_sbj_labels': gt_sub_labels,
    #                        'gt_obj_boxes': gt_obj_boxes,
    #                        'gt_obj_labels': gt_obj_labels,
    #                        'gt_prd_labels': relation_idx[:, 2]
    #                        }
    #     all_results.append(img_result_dict)
