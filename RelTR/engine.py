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

def model_to_freeze(model):

    def unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    # sent parts into eval to really freeze them
    unwrap_model(model).transformer.encoder.eval()
    unwrap_model(model).backbone.eval()

    unwrap_model(model).transformer.decoder.layers[0].self_attn_entity.eval()
    unwrap_model(model).transformer.decoder.layers[0].cross_attn_entity.eval()
    unwrap_model(model).transformer.decoder.layers[0].norm1_entity.eval()
    unwrap_model(model).transformer.decoder.layers[0].norm2_entity.eval()
    unwrap_model(model).transformer.decoder.layers[0].linear1_entity.eval()
    unwrap_model(model).transformer.decoder.layers[0].linear2_entity.eval()
    unwrap_model(model).transformer.decoder.layers[0].norm3_entity.eval()

    unwrap_model(model).transformer.decoder.layers[1].self_attn_entity.eval()
    unwrap_model(model).transformer.decoder.layers[1].cross_attn_entity.eval()
    unwrap_model(model).transformer.decoder.layers[1].norm1_entity.eval()
    unwrap_model(model).transformer.decoder.layers[1].norm2_entity.eval()
    unwrap_model(model).transformer.decoder.layers[1].linear1_entity.eval()
    unwrap_model(model).transformer.decoder.layers[1].linear2_entity.eval()
    unwrap_model(model).transformer.decoder.layers[1].norm3_entity.eval()

    unwrap_model(model).transformer.decoder.layers[2].self_attn_entity.eval()
    unwrap_model(model).transformer.decoder.layers[2].cross_attn_entity.eval()
    unwrap_model(model).transformer.decoder.layers[2].norm1_entity.eval()
    unwrap_model(model).transformer.decoder.layers[2].norm2_entity.eval()
    unwrap_model(model).transformer.decoder.layers[2].linear1_entity.eval()
    unwrap_model(model).transformer.decoder.layers[2].linear2_entity.eval()
    unwrap_model(model).transformer.decoder.layers[2].norm3_entity.eval()

    unwrap_model(model).transformer.decoder.layers[3].self_attn_entity.eval()
    unwrap_model(model).transformer.decoder.layers[3].cross_attn_entity.eval()
    unwrap_model(model).transformer.decoder.layers[3].norm1_entity.eval()
    unwrap_model(model).transformer.decoder.layers[3].norm2_entity.eval()
    unwrap_model(model).transformer.decoder.layers[3].linear1_entity.eval()
    unwrap_model(model).transformer.decoder.layers[3].linear2_entity.eval()
    unwrap_model(model).transformer.decoder.layers[3].norm3_entity.eval()

    unwrap_model(model).entity_class_embed.eval()
    unwrap_model(model).entity_bbox_embed.eval()
    return model

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    #model = model_to_freeze(model)
    criterion.train()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    stats = {
        'loss' : [],
        'class_error' : [],
        'loss_bbox' : [],
        'sub_error' : [],
        'obj_error': [],
        'rel_error' : []

    }

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        if max_norm > 0:
            trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)
        optimizer.step()

        stats['loss'].append(loss_value)
        stats['class_error'].append(loss_dict_reduced.get('class_error'))
        stats['sub_error'].append(loss_dict_reduced.get('sub_error'))
        stats['loss_bbox'].append(loss_dict_reduced.get('loss_bbox'))
        stats['obj_error'].append(loss_dict_reduced.get('obj_error'))
        stats['rel_error'].append(loss_dict_reduced.get('rel_error'))

    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    #print("Averaged stats:", metric_logger)

    #return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args):
    model.eval()
    criterion.eval()

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # header = 'Test:'

    # initilize evaluator
    # # TODO merge evaluation programs
    # if args.dataset == 'vg':
    #     evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
    #     if args.eval:
    #         evaluator_list = []
    #         for index, name in enumerate(data_loader.dataset.rel_categories):
    #             if index == 0:
    #                 continue
    #             evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    #     else:
    #         evaluator_list = None
    # else:
    #     all_results = []

    #iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    #coco_evaluator = CocoEvaluator(base_ds, iou_types)
    #all_results = []

    stats = {
        'loss' : [],
        'class_error' : [],
        'loss_bbox' : [],
        'sub_error' : [],
        'obj_error': [],
        'rel_error' : []

    }

    for samples, targets in data_loader:

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        stats['loss'].append(losses_reduced_scaled.item())
        stats['class_error'].append(loss_dict_reduced.get('class_error'))
        stats['sub_error'].append(loss_dict_reduced.get('sub_error'))
        stats['loss_bbox'].append(loss_dict_reduced.get('loss_bbox'))
        stats['obj_error'].append(loss_dict_reduced.get('obj_error'))
        stats['rel_error'].append(loss_dict_reduced.get('rel_error'))
        
        #evaluate_rel_batch_oi(outputs, targets, all_results)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        #res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        #if coco_evaluator is not None:
        #    coco_evaluator.update(res)

        #task_evaluation_sg.eval_rel_results(all_results, 100, do_val=True, do_vis=False)

    # gather the stats from all processes
    #if coco_evaluator is not None:
    #    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    #if coco_evaluator is not None:
    #    coco_evaluator.accumulate()
    #    coco_evaluator.summarize()

    #stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    #if coco_evaluator is not None:
    #    if 'bbox' in postprocessors.keys():
    #        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    return stats#, coco_evaluator

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)
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