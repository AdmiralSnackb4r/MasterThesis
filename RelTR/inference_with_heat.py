# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import torch
import torchvision.transforms as T
from models import build_model
from visualization_techniques import plot_backbone, attention_rollout
import networkx as nx
import torch.nn as nn
import torch.nn.init as init
import math
from collections import defaultdict

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train_extra\\bayreuth\\bayreuth_000000_000003_leftImg8bit.png',
                          help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='F:\\scenario_runner-0.9.15\\Data\\_out\\FollowLeadingVehicle_1\\rgb\\filtered\\00006544.png',
    #                     help="Path of the test image")
    # parser.add_argument('--img_path', type=str, default='demo/cat.jpg',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\BDD100\\bdd100k_images_10k\\10k\\val\\ab309345-00000000.jpg',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\hamburg\\hamburg_000000_021961_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\Mappillary\\training\\images\\0gFsMvCekBJRwPYYl_k12Q.jpg',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\test\\munich\\munich_000193_000019_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\06_images\\images\\14017.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\zurich\\zurich_000061_000019_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='F:\\scenario_runner-0.9.15\\Data\\_out\\VehicleTurningRight_7\\rgb\\filtered\\00008826.png',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='F:\\scenario_runner-0.9.15\\Data\\_out\\OppositeVehicleRunningRedLight_5_3\\rgb\\filtered\\00002064.png',
    #                     help="Path of the test image")
    

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

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt\\run_full_sim_from_sim_and_real_enfcoder\\checkpoint_re_0539.pth', help='resume from checkpoint')
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


def box_center(box):
    """Calculate center (cx, cy) of a bounding box."""
    x_min, y_min, x_max, y_max = box
    return [(x_min + x_max) / 2, (y_min + y_max) / 2]

def euclidean_dist(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

#def merge_boxes(box1, box2):
#    """Average the coordinates of two boxes."""
#    return [(b1 + b2) / 2 for b1, b2 in zip(box1, box2)]

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

def merge_predictions(net1_boxes, net2_boxes, net1_labels, net2_labels):
    result = []

    for i, box2 in enumerate(net2_boxes):
        label2 = net2_labels[i]
        box2_center = box_center(box2)

        # Filter net1 boxes by same label
        candidate_indices = [j for j, lbl in enumerate(net1_labels) if lbl == label2]

        # Further filter only boxes that overlap (iou > 0) with box2
        overlapping_indices = []
        for j in candidate_indices:
            box1 = net1_boxes[j]
            if iou(box1, box2) > 0:
                overlapping_indices.append(j)

        # If none overlap, fallback: use all candidates with same label
        if not overlapping_indices:
            overlapping_indices = candidate_indices

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

def check_frozen_weights_integrity(model, checkpoint_path, atol=1e-5, rtol=1e-5):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_state_dict = checkpoint['model']

    # Remove 'module.' prefix if present in checkpoint
    cleaned_checkpoint = {
        k.replace('module.', ''): v for k, v in checkpoint_state_dict.items()
    }

    current_state_dict = model.state_dict()
    
    matched = []
    mismatched = []
    missing_in_model = []

    for name, ckpt_param in cleaned_checkpoint.items():
        if name in current_state_dict:
            model_param = current_state_dict[name]
            if torch.allclose(model_param, ckpt_param, atol=atol, rtol=rtol):
                matched.append(name)
                print(f"[MATCH] {name} -- Values accurate")
            else:
                mismatched.append(name)
                print(f"[MISMATCH] {name} - Values differ.")
        else:
            missing_in_model.append(name)
            print(f"[MISSING] {name} - Not found in model.")

    print(f"\n✅ [CHECK COMPLETE]")
    print(f"Matched: {len(matched)}")
    print(f"Mismatched: {len(mismatched)}")
    print(f"Missing: {len(missing_in_model)}")

    return {
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing_in_model,
    }

def merge_model(full_model, first_part, output_log="merge_log.txt"):
    full_checkpoint = torch.load(full_model, weights_only=False)
    first_part_checkpoint = torch.load(first_part, weights_only=False)

    new_state_dict = {}
    for k, v in first_part_checkpoint['model'].items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    log_lines = []

    for key in full_checkpoint['model']:
        if key in new_state_dict:
            full_checkpoint['model'][key] = new_state_dict[key]
            log_lines.append(f"{key}: REPLACED from first_part_checkpoint")
        else:
            log_lines.append(f"{key}: KEPT from full_checkpoint")

    # Save log to file
    with open(output_log, "w") as f:
        for line in log_lines:
            f.write(line + "\n")
    
    print(f"Merge completed. Log saved to {output_log}")
    return full_checkpoint  # Optional, if you want to save/use it afterwards

def initialize_and_freeze(model, first_part_checkpoint_path):
    checkpoint = torch.load(first_part_checkpoint_path, map_location='cpu', weights_only=False)
    pretrained_state_dict = checkpoint['model']

    # Clean the state dict (remove 'module.' if present)
    cleaned_state_dict = {}
    for k, v in pretrained_state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state_dict[new_key] = v

    # Load the cleaned weights into the model (strict=False allows partial load)
    model.load_state_dict(cleaned_state_dict, strict=False)

    # Track how many matched weights we found
    matched, unmatched = [], []

    for name, param in model.named_parameters():
        if name in cleaned_state_dict:
            if torch.allclose(param.data, cleaned_state_dict[name], atol=1e-5):
                matched.append(name)
                param.requires_grad = False
                print(f"[FREEZED] {name} - matched and frozen")
            else:
                unmatched.append(name)
                print(f"[WARNING] {name} found but values don't match closely")
        else:
            param.requires_grad = True
            print(f"[TRAINABLE] {name} - not in checkpoint")

    # Optional: initialize only trainable parameters
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if m.weight.requires_grad:
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None and m.bias.requires_grad:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.weight.requires_grad:
                init.constant_(m.weight, 1.0)
            if m.bias.requires_grad:
                init.constant_(m.bias, 0)

    model.apply(init_weights)

    print(f"\n[SUMMARY] {len(matched)} weights matched and frozen.")
    print(f"[SUMMARY] {len(unmatched)} weights were in checkpoint but had mismatched values.")

    return model



def main(args):

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # Carla classes
    CLASSES = [ 'N/A', 'ground', 'road', 'side walk', 'bridge', 'pole', 'traffic light', 'traffic sign', 'person', 'car', 'truck', 'bicycle']
    #CLASSES = [ 'road', 'side walk', 'ground', 'pole', 'traffic light', 'traffic sign', 'person', 'car', 'truck', 'bicycle', 'bridge']

    # 2 -> 1
    # 0 -> 2
    # 1 -> 0
    # 8 -> 7
    # 6 -> 5
    # 4 -> 3
    # 5 -> 4
    # 7 -> 6
    # 9 -> 8
    # 10 -> 9
    # 3 -> 10

    REL_CLASSES = ['__background__', 'on', 'attached to', 'on right side of', 'parking on', 'on left side of', 'same road line as', 'on right lane of', 'on opposing side of', 'on left lane of', 'driving from right to left', 'driving from left to right', 'on middle lane of',
                   'infront of', 'behind', 'riding', 'next to', 'turning right on', 'driving on', 'turning left on', 'is hitting']
    
    #REL_CLASSES = ['on', 'attached to', 'on right side of', 'parking on', 'on left side of', 'same road line as', 'on right lane of', 'on opposing side of', 'on left lane of', 'driving from left to right', 'driving from right to left', 'on middle lane of',
    #               'infront of', 'behind', 'riding', 'next to', 'turning right on', 'driving on', 'turning left on', 'is hitting']
    # 9 <-> 10

     # VG classes
    # CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
    #             'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
    #             'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
    #             'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
    #             'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
    #             'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
    #             'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
    #             'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
    #             'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
    #             'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
    #             'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
    #             'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
    #             'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
    #             'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    # REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
    #             'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
    #             'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
    #             'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
    #             'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
    #             'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']


    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume, weights_only=False)
    #result = check_frozen_weights_integrity(model, 'ckpt\\run_first_part\\checkpoint0248_.pth')
    #ckpt = merge_model(full_model='ckpt\\run_full_sim\\checkpoint0199.pth', first_part='ckpt\\run_first_part\\checkpoint0248_.pth')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt

    #print(result)
    #exit()

# Count number of parameters
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"Total parameters in checkpoint: {total_params:,}")
    model.load_state_dict(ckpt['model'])
    #model = initialize_and_freeze(model, first_part_checkpoint_path='ckpt\\run_first_part\\checkpoint0248_.pth')
    model.eval()

    img_path = args.img_path
    im = Image.open(img_path).convert("RGB")

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.+ confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_or(probas.max(-1).values > 0.3, torch.logical_or(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))

    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    topk = 100
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]

    # use lists to store the outputs via up-values
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
    dec_attn_weights_sub_list = [] # To store sub attention weights from ALL decoder layers
    dec_attn_weights_obj_list = [] # To store obj attention weights from ALL decoder layers

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        )#,
        #model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
        #    lambda self, input, output: dec_attn_weights_sub.append(output[1])
        #),
        #model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
        #    lambda self, input, output: dec_attn_weights_obj.append(output[1])
        #)
    ]

    # Register hooks for ALL decoder layers to collect attention weights
    # 'model.transformer.decoder.layers' is typically a ModuleList
    for i, layer in enumerate(model.transformer.decoder.layers):
        hooks.append(
            layer.cross_attn_sub.register_forward_hook(
                lambda self, input, output: dec_attn_weights_sub_list.append(output[1])
            )
        )
        hooks.append(
            layer.cross_attn_obj.register_forward_hook(
                lambda self, input, output: dec_attn_weights_obj_list.append(output[1])
            )
        )

    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        #dec_attn_weights_sub = dec_attn_weights_sub[0]
        #dec_attn_weights_obj = dec_attn_weights_obj[0]

        avg_dec_attn_weights_sub = torch.stack(dec_attn_weights_sub_list, dim=0).mean(dim=0)
        avg_dec_attn_weights_obj = torch.stack(dec_attn_weights_obj_list, dim=0).mean(dim=0)

        subject_filter = None  # e.g., "car" to filter only "car" subjects
        object_filter = None  # e.g., "traffic light" to filter only "traffic light" object

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]
        im_w, im_h = im.size
        vis = []

        fig, axs = plt.subplots(ncols=20, nrows=3, figsize=(22, 7))
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            
            subj_cls = CLASSES[probas_sub[idx].argmax()]
            obj_cls = CLASSES[probas_obj[idx].argmax()]


            # Apply subject and object filters
            if subject_filter and subj_cls != subject_filter:
                ax_i_tmp = ax_i
                continue
            else:
                ax_i_tmp = None
            if object_filter and obj_cls != object_filter:
                ax_i_tmp = ax_i
                continue
            else:
                ax_i_tmp = None

            print(ax_i)

            #idx_plot = idx
            #if idx_plot == 3 :
            #    idx_plot = 1

            if ax_i_tmp is not None:
                ax_i = ax_i_tmp

            img_up = avg_dec_attn_weights_sub[0, idx].view(h, w)
            title_up = f'query id: {idx.item()}'
            img_down = avg_dec_attn_weights_obj[0, idx].view(h, w)
            img_rgb = im
            patch_up = plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                       fill=False, color='blue', linewidth=2.5)
            patch_down = plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                       fill=False, color='orange', linewidth=2.5)
            title = f"{CLASSES[probas_sub[idx].argmax()]} {REL_CLASSES[probas[idx].argmax()]} {CLASSES[probas_obj[idx].argmax()]}"

            vis.append((img_up, title_up, img_down, img_rgb, patch_up, patch_down, title))

            # ax = ax_i[0]
            # ax.imshow(avg_dec_attn_weights_sub[0, idx].view(h, w))
            # ax.axis('off')
            # ax.set_title(f'query id: {idx.item()}')
            # ax = ax_i[1]
            # ax.imshow(avg_dec_attn_weights_obj[0, idx].view(h, w))
            # ax.axis('off')
            # ax = ax_i[2]
            # ax.imshow(im)
            # ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
            #                            fill=False, color='blue', linewidth=2.5))
            # ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
            #                            fill=False, color='orange', linewidth=2.5))

            # ax.axis('off')
            # ax.set_title(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()])




        fig, axs = plt.subplots(ncols=10, nrows=3, figsize=(22, 7))
        for idx, ax_i, vi in \
                zip(keep_queries, axs.T, vis):
            ax = ax_i[0]
            ax.imshow(vi[0])
            ax.axis('off')
            ax.set_title(vi[1])
            ax = ax_i[1]
            ax.imshow(vi[2])
            ax.axis('off')
            ax = ax_i[2]
            ax.imshow(vi[3])
            ax.add_patch(vi[4])
            ax.add_patch(vi[5])

            ax.axis('off')
            ax.set_title(vi[6], fontsize=10)

        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)