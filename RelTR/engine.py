# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import numpy as np

import torch

from datasets.coco_eval import CocoEvaluator
import util.misc as utils
from util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from lib.openimages_evaluation import task_evaluation_sg

def model_to_freeze(model, freeze_entity_layers=6):
    def unwrap_model(m):
        return m.module if hasattr(m, "module") else m

    model = unwrap_model(model)

    # Set encoder and backbone to eval
    model.transformer.encoder.eval()
    model.backbone.eval()

    # Set decoder entity-specific submodules to eval
    for i in range(freeze_entity_layers):
        layer = model.transformer.decoder.layers[i]
        entity_modules = [
            layer.self_attn_entity,
            layer.cross_attn_entity,
            layer.norm1_entity,
            layer.norm2_entity,
            layer.linear1_entity,
            layer.linear2_entity,
            layer.norm3_entity,
        ]
        for module in entity_modules:
            module.eval()

    # Set entity head modules to eval
    model.entity_class_embed.eval()
    model.entity_bbox_embed.eval()

    return model

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    model = model_to_freeze(model_to_freeze, 6)
    criterion.train()

    stats = {
        'loss': [],
        'class_error': [],
        'loss_bbox': [],
        'sub_error': [],
        'obj_error': [],
        'rel_error': []
    }

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        # Logging-safe loss
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_value = sum(loss_dict_scaled.values()).item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        losses.backward()

        if max_norm > 0:
            trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)

        optimizer.step()

        # Collect stats
        stats['loss'].append(loss_value)
        for key in ['class_error', 'sub_error', 'loss_bbox', 'obj_error', 'rel_error']:
            stats[key].append(loss_dict_reduced.get(key))

    return stats


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args):
    model.eval()
    criterion.eval()

    stats = {
        'loss': [],
        'class_error': [],
        'loss_bbox': [],
        'sub_error': [],
        'obj_error': [],
        'rel_error': []
    }

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_value = sum(loss_dict_scaled.values()).item()

        # Record stats
        stats['loss'].append(loss_value)
        for key in ['class_error', 'sub_error', 'loss_bbox', 'obj_error', 'rel_error']:
            stats[key].append(loss_dict_reduced.get(key))

        # Convert outputs to final predictions
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # Optional: COCO or custom evaluator logic can go here

    return stats

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=2)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=2)
        rel_scores = outputs['rel_logits'][batch][:,1:-1].softmax(-1)

        pred_entry = {'sub_boxes': sub_bboxes_scaled,
                      'sub_classes': pred_sub_classes.cpu().clone().numpy(),
                      'sub_scores': pred_sub_scores.cpu().clone().numpy(),
                      'obj_boxes': obj_bboxes_scaled,
                      'obj_classes': pred_obj_classes.cpu().clone().numpy(),
                      'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                      'rel_scores': rel_scores.cpu().clone().numpy()}

        evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)


def evaluate_rel_batch_oi(outputs, targets, all_results):

    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

        rel_scores = outputs['rel_logits'][batch][:, :-1].softmax(-1)

        relation_idx = target['rel_annotations'].cpu().numpy()
        gt_sub_boxes = target_bboxes_scaled[relation_idx[:, 0]]
        gt_sub_labels = target['labels'][relation_idx[:, 0]].cpu().clone().numpy()
        gt_obj_boxes = target_bboxes_scaled[relation_idx[:, 1]]
        gt_obj_labels = target['labels'][relation_idx[:, 1]].cpu().clone().numpy()

        img_result_dict = {'sbj_boxes': sub_bboxes_scaled,
                           'sbj_labels': pred_sub_classes.cpu().clone().numpy(),
                           'sbj_scores': pred_sub_scores.cpu().clone().numpy(),
                           'obj_boxes': obj_bboxes_scaled,
                           'obj_labels': pred_obj_classes.cpu().clone().numpy(),
                           'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                           'prd_scores': rel_scores.cpu().clone().numpy(),
                           'image': str(target['image_id'].item())+'.jpg',
                           'gt_sbj_boxes': gt_sub_boxes,
                           'gt_sbj_labels': gt_sub_labels,
                           'gt_obj_boxes': gt_obj_boxes,
                           'gt_obj_labels': gt_obj_labels,
                           'gt_prd_labels': relation_idx[:, 2]
                           }
        all_results.append(img_result_dict)