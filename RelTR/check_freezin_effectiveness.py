import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import torch
import torchvision.transforms as T
from models import build_model, custom_build_model
from visualization_techniques import plot_backbone, attention_rollout
import networkx as nx
import torch.nn as nn
import torch.nn.init as init
import math
from collections import OrderedDict
#import torchvision.transforms.functional as T
import torch.nn.functional as F

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train_extra\\bayreuth\\bayreuth_000000_000831_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='F:\\scenario_runner-0.9.15\\Data\\_out\\DynamicObjectCrossing_1_3\\rgb\\filtered\\00048940.png',
    #                     help="Path of the test image")
    # parser.add_argument('--img_path', type=str, default='demo/cat.jpg',
    #                     help="Path of the test image")
    parser.add_argument('--img_path', type=str, default='S:\\Datasets\\BDD100\\bdd100k_images_10k\\10k\\test\\d6f2e089-8a310d73.jpg',
                         help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\cologne\\cologne_000082_000019_leftImg8bit.png',
    #                      help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
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
    parser.add_argument('--full_checkpoint', default='ckpt\\run_full_freezed\\checkpoint0199.pth', help='resume from checkpoint')
    parser.add_argument('--first_part_checkpoint', default='ckpt\\run_first_part\\checkpoint0248_.pth', help='resume from checkpoint')
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



# Global dicts to store hooked outputs
hook_outputs_full = {}
hook_outputs_first = {}

def get_activation_hook(storage_dict, name):
    def hook(module, input, output):
        # If output is a tuple, process each tensor
        if isinstance(output, tuple):
            processed = [o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output]
            storage_dict[name] = processed
        else:
            storage_dict[name] = output.detach().cpu()
    return hook

def register_hooks(model, storage_dict):
    hooks = []
    hooks.append(model.backbone.register_forward_hook(get_activation_hook(storage_dict, "backbone")))
    hooks.append(model.transformer.encoder.layers[0].register_forward_hook(get_activation_hook(storage_dict, "encoder_layer0")))
    hooks.append(model.transformer.decoder.layers[0].self_attn_entity.register_forward_hook(get_activation_hook(storage_dict, "self_attn_entity")))
    hooks.append(model.transformer.decoder.layers[0].cross_attn_entity.register_forward_hook(get_activation_hook(storage_dict, "cross_attn_entity")))
    return hooks

def compute_similarity(name, out1, out2):
    if isinstance(out1, list) or isinstance(out1, tuple):
        for i, (t1, t2) in enumerate(zip(out1, out2)):
            if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
                continue
            flat1 = t1.view(-1).float()
            flat2 = t2.view(-1).float()
            cos_sim = F.cosine_similarity(flat1, flat2, dim=0)
            mse = F.mse_loss(flat1.unsqueeze(0), flat2.unsqueeze(0))
            print(f"{name}[{i}]".ljust(30), f"| MSE: {mse:.6e} | Cosine Similarity: {cos_sim:.6f}")
    else:
        flat1 = out1.view(-1).float()
        flat2 = out2.view(-1).float()
        cos_sim = F.cosine_similarity(flat1, flat2, dim=0)
        mse = F.mse_loss(flat1.unsqueeze(0), flat2.unsqueeze(0))
        print(f"{name}".ljust(30), f"| MSE: {mse:.6e} | Cosine Similarity: {cos_sim:.6f}")

def main(args):
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Build models
    full_model, _, _ = build_model(args)
    first_part_model, _, _ = custom_build_model(args)

    # Load checkpoints
    ckpt_full = torch.load(args.full_checkpoint, weights_only=False, map_location='cpu')
    ckpt_first_part = torch.load(args.first_part_checkpoint, weights_only=False, map_location='cpu')

    full_model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in ckpt_full['model'].items()}, strict=True)
    first_part_model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in ckpt_first_part['model'].items()}, strict=True)

    full_model.eval()
    first_part_model.eval()

    print(f"Models loaded from:\n- Full:       {args.full_checkpoint}\n- First Part: {args.first_part_checkpoint}")

    # Register hooks
    register_hooks(full_model, hook_outputs_full)
    register_hooks(first_part_model, hook_outputs_first)

    # Prepare input
    img = Image.open(args.img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

    # Run forward pass with no grad
    with torch.no_grad():
        _ = full_model(img_tensor)
        _ = first_part_model(img_tensor)

    # Compare hooked module outputs
    for name in ["backbone", "encoder_layer0", "self_attn_entity", "cross_attn_entity"]:
        if name in hook_outputs_full and name in hook_outputs_first:
            compute_similarity(name, hook_outputs_first[name], hook_outputs_full[name])
        else:
            print(f"{name:25s} | Output not captured.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Freeze Checker', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)