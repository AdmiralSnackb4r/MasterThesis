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
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as TF
import torchvision.models as models

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train_extra\\bayreuth\\bayreuth_000000_000003_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path_carla', type=str, default='F:\\scenario_runner-0.9.15\\Data\\_out\\OppositeVehicleRunningRedLight_5_3\\rgb\\filtered\\00002064.png',
    #                     help="Path of the test image")
    # parser.add_argument('--img_path', type=str, default='demo/cat.jpg',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\BDD100\\bdd100k_images_10k\\10k\\test\\d1b624d3-00000000.jpg',
    #                     help="Path of the test image")
    parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\cologne\\cologne_000082_000019_leftImg8bit.png',
                          help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\Mappillary\\training\\images\\0gFsMvCekBJRwPYYl_k12Q.jpg',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train_extra\\bayreuth\\bayreuth_000000_000834_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\aachen\\aachen_000157_000019_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\zurich\\zurich_000061_000019_leftImg8bit.png',
    #                      help="Path of the test image")       #left side debatable
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\zurich\\zurich_000009_000019_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\jena\\jena_000001_000019_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\jena\\jena_000099_000019_leftImg8bit.png',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\jena\\jena_000105_000019_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\jena\\jena_000056_000019_leftImg8bit.png',
    #                      help="Path of the test image")       #gone wrong teribly
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\zurich\\zurich_000008_000019_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\zurich\\zurich_000009_000019_leftImg8bit.png',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\dusseldorf\\dusseldorf_000206_000019_leftImg8bit.png',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\test\\munich\\munich_000008_000019_leftImg8bit.png',
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
    parser.add_argument('--resume', default='ckpt\\run_full_sim_from_sim_and_real\\checkpoint_re_0619.pth', help='resume from checkpoint')
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
            if iou(box1, box2) > 0.3:
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


def style_transfer(content_path, style_path, image_size=600, steps=200, style_weight=1e10, content_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Bild laden und vorbereiten
    def load_image(img_path):
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]),  # RGB
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)
        return image.to(device)

    content = load_image(content_path)
    style = load_image(style_path)

    # VGG19 laden
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    # Zielbild
    target = nn.Parameter(content.clone().detach())

    # Layer definieren
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # Feature Extraction
    def get_features(x, model, layers):
        features = {}
        i = 0
        for layer in model.children():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f"conv_{i}"
                if name in layers:
                    features[name] = x
        return features

    # Gram-Matrix
    def gram_matrix(tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        return torch.mm(tensor, tensor.t()) / (c * h * w)

    # Features
    content_features = get_features(content, vgg, content_layers)
    content_features = {k: v.detach() for k, v in content_features.items()}
    style_features = get_features(style, vgg, style_layers)
    style_grams = {layer: gram_matrix(style_features[layer].detach()) for layer in style_features}

    # Optimierer
    optimizer = optim.Adam([target], lr=0.003)

    # Optimierung
    for step in range(steps):
        target_features = get_features(target, vgg, content_layers + style_layers)

        content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4']) ** 2)
        style_loss = 0
        for layer in style_layers:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += torch.mean((target_gram - style_gram) ** 2)
        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Tensor zurückwandeln in Bild
    def im_convert(tensor):
        image = tensor.to("cpu").clone().detach()
        image = image.squeeze(0)
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        image = image.clamp(0, 1)
        image = TF.to_pil_image(image)
        return image
    
    #image = TF.to_pil_image(image)

    return im_convert(target)


def main(args):

    transform = T.Compose([
        T.Resize(1333),
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

    REL_CLASSES = [ '__background__', 'on', 'attached to', 'on right side of', 'parking on', 'on left side of', 'same road line as', 'on right lane of', 'on opposing side of', 'on left lane of', 'driving from right to left', 'driving from left to right', 'on middle lane of',
                   'infront of', 'behind', 'riding', 'next to', 'turning right on', 'driving on', 'turning left on', 'is hitting']


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


    #print(result)
    #exit()

# Count number of parameters

    #img_path = args.img_path
    #im = style_transfer(args.img_path, args.img_path_carla)
    im = Image.open(args.img_path).convert("RGB")

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume, weights_only=False)
    #result = check_frozen_weights_integrity(model, 'ckpt\\run_first_part\\checkpoint0248_.pth')
    #ckpt = merge_model(full_model='ckpt\\run_full_sim\\checkpoint0199.pth', first_part='ckpt\\run_first_part\\checkpoint0248_.pth')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt

    total_params = sum(v.numel() for v in state_dict.values())
    print(f"Total parameters in checkpoint: {total_params:,}")
    model.load_state_dict(ckpt['model'])
    #model = initialize_and_freeze(model, first_part_checkpoint_path='ckpt\\run_first_part\\checkpoint0248_.pth')
    model.eval()

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)



    # Compute softmax probabilities for various logits
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]     # [num_queries, num_rel_classes]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1] # [num_queries, num_sub_classes]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1] # [num_queries, num_obj_classes]
    probas_entity = outputs['pred_logits'].softmax(-1)[0, :, :-1] # [num_queries, num_entity_classes]

    # Keep entity predictions with confidence > 0.85
    keep_entity = probas_entity.max(-1).values > 0.85  # boolean mask [num_queries]
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep_entity], im.size)  # filtered boxes
    probas_entity_kept = probas_entity[keep_entity]
    labels_entity = probas_entity_kept.argmax(dim=1)

    # Select top-k confident entities
    topk = 300
    keep_queries_entity = torch.nonzero(keep_entity, as_tuple=True)[0]
    indices_entity = torch.argsort(-probas_entity_kept.max(-1)[0])[:topk]
    keep_queries_entity = keep_queries_entity[indices_entity]

    # Keep relation queries with confidence > 0.5 in all three logits
    keep = torch.logical_and(probas.max(-1).values > 0.7,
            torch.logical_and(probas_sub.max(-1).values > 0.7, probas_obj.max(-1).values > 0.7))

    # Filter labels and boxes for subjects and objects accordingly
    labels_sub = probas_sub[keep].argmax(dim=1)
    labels_obj = probas_obj[keep].argmax(dim=1)

    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    #Merge boxes based on labels (you need your merge_predictions updated accordingly)
    sub_bboxes_scaled = merge_predictions(
        bboxes_scaled, sub_bboxes_scaled, labels_entity, labels_sub)#

    obj_bboxes_scaled = merge_predictions(
        bboxes_scaled, obj_bboxes_scaled, labels_entity, labels_obj)

    # Select top 10 relation queries based on combined confidence score
    topk = 20
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    scores = probas[keep].max(-1)[0] * probas_sub[keep].max(-1)[0] * probas_obj[keep].max(-1)[0]
    indices = torch.argsort(-scores)[:topk]
    keep_queries = keep_queries[indices]


    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights_entity, dec_attn_weights_sub, dec_attn_weights_obj = [], [], [], [], []


    print(len(model.backbone))

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_entity.register_forward_hook(
            lambda self, input, output: dec_attn_weights_entity.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )
    ]
    with torch.no_grad():
        # propagate through the model
        #outputs = model(img)

        for hook in hooks:
            hook.remove()

        #print(conv_features)

        #for i in conv_features:
        #    print(i)


        enc_attn_weights_list = enc_attn_weights
        dec_attn_weights_entity_list = dec_attn_weights_entity
        dec_attn_weights_sub_list = dec_attn_weights_sub
        dec_attn_weights_obj_list = dec_attn_weights_obj


        # # don't need the list anymore
        # conv_features = conv_features[0]
        # enc_attn_weights = enc_attn_weights[0]
        # dec_attn_weights_entity = dec_attn_weights_entity[0]
        # dec_attn_weights_sub = dec_attn_weights_sub[0]
        # dec_attn_weights_obj = dec_attn_weights_obj[0]

        #attention_rollout(enc_attn_weights)

        # --------------------------------------
        # Step 1: Create unique colors
        # --------------------------------------

        # Number of unique objects (subjects + objects)
        num_entities = len(keep_queries) * 2  # each query gives 1 subject + 1 object
        cmap = cm.get_cmap('tab20', num_entities)  # or use 'tab20', 'hsv', etc.

        # Assign a unique color index for each subject/object
        entity_color_map = {}  # maps node_name -> color

        # Create figure: 1 row, 2 columns
        fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(22, 12))

        # --------------------------------------
        # Step 2: Plot left image with colored bounding boxes and labels
        # --------------------------------------
        ax_left.imshow(im)
        ax_left.axis('off')
        ax_left.set_title('Detected Objects', fontsize=16)

        # Counter for assigning colors
        color_idx = 0

        # for idx, (bxmin, bymin, bxmax, bymax) in zip(keep_queries_entity, bboxes_scaled[indices_entity]):
        #     entity_name = CLASSES[probas_entity[idx].argmax()]

        #     entity_name = f'Ent-{idx}: {entity_name}'
        #     rect_e = plt.Rectangle(
        #         (bxmin, bymin), width=bxmax - bxmin, height=bymax - bymin,
        #         fill=False, edgecolor='blue', linewidth=3
        #     )
        #     ax_left.add_patch(rect_e)
        #     ax_left.text(
        #         bxmin, bymin - 5, entity_name,
        #         fontsize=10, color='blue',
        #         verticalalignment='bottom', weight='bold',
        #         bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3', alpha=0.7)
        #     )


        def round_box(box, precision=2):
            return tuple(round(float(v), precision) for v in box)

        # Step 1: Create unique subject and object box maps
        unique_subj_boxes = {}
        unique_obj_boxes = {}

        subj_box_to_node = {}
        obj_box_to_node = {}

        subject_filter = None  # e.g., "car" to filter only "car" subjects
        object_filter = None   # e.g., "traffic light" to filter only "traffic light" object

        duplicate_subj_indices = defaultdict(list)
        duplicate_obj_indices = defaultdict(list)

        for idx, sub_box in zip(keep_queries, sub_bboxes_scaled[indices]):
            idx = int(idx)
            box_key = round_box(sub_box)
            if box_key not in unique_subj_boxes:
                subj_cls = CLASSES[probas_sub[idx].argmax()]
                if subject_filter is not None and subj_cls != subject_filter:
                    continue
                print(sub_box, box_key)
                node_name = f"{subj_cls}-X:{box_key[0]}-Y:{box_key[2]}"
                unique_subj_boxes[box_key] = node_name
            subj_box_to_node[idx] = unique_subj_boxes[box_key]
            duplicate_subj_indices[unique_subj_boxes[box_key]].append(idx)

        for idx, obj_box in zip(keep_queries, obj_bboxes_scaled[indices]):
            idx = int(idx)
            box_key = round_box(obj_box)
            if box_key not in unique_obj_boxes:
                obj_cls = CLASSES[probas_obj[idx].argmax()]
                if object_filter is not None and obj_cls != object_filter:
                    continue
                node_name = f"{obj_cls}-X:{box_key[0]}-Y:{box_key[2]}"
                unique_obj_boxes[box_key] = node_name
            obj_box_to_node[idx] = unique_obj_boxes[box_key]
            duplicate_obj_indices[unique_obj_boxes[box_key]].append(idx)

        # Step 2: Draw subject and object bounding boxes (only first per group)
        for idx in keep_queries:
            idx = int(idx)  # 🔥 Fix tensor key issue
            if idx not in subj_box_to_node or idx not in obj_box_to_node:
                continue
            subj_node = subj_box_to_node[idx]
            obj_node = obj_box_to_node[idx]

            subj_color = entity_color_map.setdefault(subj_node, cmap(color_idx)); color_idx += 1
            obj_color = entity_color_map.setdefault(obj_node, cmap(color_idx)); color_idx += 1

            subj_box_key = [k for k, v in unique_subj_boxes.items() if v == subj_node][0]
            obj_box_key = [k for k, v in unique_obj_boxes.items() if v == obj_node][0]

            if idx == duplicate_subj_indices[subj_node][0]:
                sxmin, symin, sxmax, symax = subj_box_key
                subj_cls = subj_node.split('-')[0]
                rect_s = plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                    fill=False, edgecolor=subj_color, linewidth=3)
                ax_left.add_patch(rect_s)
                ax_left.text(sxmin, symin - 5, subj_cls, fontsize=10, color=subj_color,
                            verticalalignment='bottom', weight='bold',
                            bbox=dict(facecolor='white', edgecolor=subj_color, boxstyle='round,pad=0.3', alpha=0.7))

            if idx == duplicate_obj_indices[obj_node][0]:
                oxmin, oymin, oxmax, oymax = obj_box_key
                obj_cls = obj_node.split('-')[0]
                rect_o = plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                    fill=False, edgecolor=obj_color, linewidth=3)
                ax_left.add_patch(rect_o)
                ax_left.text(oxmin, oymin - 5, obj_cls, fontsize=10, color=obj_color,
                            verticalalignment='bottom', weight='bold',
                            bbox=dict(facecolor='white', edgecolor=obj_color, boxstyle='round,pad=0.3', alpha=0.7))

        # Group relationships between same subj→obj pairs
        edge_relation_map = defaultdict(list)

        for idx in keep_queries:
            idx = int(idx)
            if idx not in subj_box_to_node or idx not in obj_box_to_node:
                continue
            subj_node = subj_box_to_node[idx]
            obj_node = obj_box_to_node[idx]
            rel_cls = REL_CLASSES[probas[idx].argmax()]

            edge_relation_map[(subj_node, obj_node)].append(rel_cls)

        # Step 4: Draw graph with aggregated relationship labels
        G = nx.DiGraph()

        for (subj_node, obj_node), rels in edge_relation_map.items():
            print(subj_node)
            G.add_node(subj_node)
            G.add_node(obj_node)

            # Combine unique relationships into a single label string
            combined_rel = " / ".join(sorted(set(rels)))
            G.add_edge(subj_node, obj_node, label=combined_rel)

        # Get node colors
        node_colors = [entity_color_map[node] for node in G.nodes()]
        pos = nx.spring_layout(G, k=1.2, iterations=50)

        # Draw nodes and edges
        nx.draw(
            G, pos, ax=ax_right, with_labels=True,
            node_color=node_colors, node_size=2000,
            font_size=10, font_weight='bold', arrowsize=20,
            edgecolors='black'
        )

        # Draw edge labels (multiple relationships per edge)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_color='blue', font_size=10
        )

        ax_right.set_title('Relationships Graph', fontsize=16)
        ax_right.axis('off')


        # --------------------------------------
        # Step 4: Final layout
        # --------------------------------------
        fig.tight_layout()
        plt.savefig('pretrained_inference.png', dpi=600, bbox_inches='tight')
        plt.show()



        # # get the feature map shape
        # h, w = conv_features['0'].tensors.shape[-2:]
        # im_w, im_h = im.size

        # print(conv_features['0'].tensors.shape)
        # #plot_backbone(conv_features['0'].tensors, im)

        # print(type(dec_attn_weights_sub))  # Check if it's a tuple
        # print(len(dec_attn_weights_sub))   # Check its length

        # fig, axs = plt.subplots(ncols=len(indices), nrows=5, figsize=(22, 7))
        # for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
        #         zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
        #     ax = ax_i[0]
        #     ax.axis('off')
        #     ax.set_title(f'conv features')
        #     ax.imshow(conv_features['0'].tensors[0, idx].view(h, w).cpu())
        #     ax = ax_i[1]
        #     ax.set_title(f'entity attention')
        #     ax.imshow(enc_attn_weights[0, idx].view(h, w))
        #     ax = ax_i[2]
        #     ax.set_title(f'subject attention')
        #     ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
        #     ax = ax_i[3]
        #     ax.set_title(f'object attention')
        #     ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
        #     ax.axis('off')
        #     ax = ax_i[4]
        #     ax.imshow(im)
        #     ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
        #                                fill=False, color='blue', linewidth=2.5))
        #     ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
        #                                fill=False, color='orange', linewidth=2.5))

        #     ax.axis('off')
        #     ax.set_title(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()], fontsize=10)

        # fig.tight_layout()
        # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
