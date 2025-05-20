# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_carla_dataset, get_coco_api_from_dataset
import datasets.transforms as T
from engine import evaluate, train_one_epoch
from models import build_model
from distributed_utils import *
from collections import OrderedDict

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer") #6
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer") #6
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
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--ann_path', default='./data/vg/', type=str)
    parser.add_argument('--img_folder', default='/home/cong/Dokumente/tmp/data/visualgenome/images/', type=str)
    parser.add_argument('--datapath', default='/p/scratch/hai_1008/kromm3/Carla/Data', type=str)

    parser.add_argument('--output_dir', default='/p/scratch/hai_1008/kromm3/RelTR/ckpt/run_full_sim_1',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser

def initialize_and_freeze(model, first_part_checkpoint_path):
    checkpoint = torch.load(first_part_checkpoint_path, map_location='cpu', weights_only = False)
    pretrained_state_dict = checkpoint['model']

    cleaned_state_dict = {}
    for k, v in pretrained_state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict, strict=False)

    for name, param in model.named_parameters():
        if name in cleaned_state_dict:
            param.requires_grad = False
        else:
            param.requires_grad = True

    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if m.weight.requires_grad:
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None and m.bias.requires_grad:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias.requires_grad:
                init.constant_(m.bias, 0)
            if m.weight.requires_grad:
                init.constant_(m.weight, 1.0)

    model.apply(init_weights)

    # sent parts into eval to really freeze them
    model.transformer.encoder.eval()
    model.backbone.eval()

    model.transformer.decoder.layers[0].self_attn_entity.eval()
    model.transformer.decoder.layers[0].cross_attn_entity.eval()
    model.transformer.decoder.layers[0].norm1_entity.eval()
    model.transformer.decoder.layers[0].norm2_entity.eval()
    model.transformer.decoder.layers[0].linear1_entity.eval()
    model.transformer.decoder.layers[0].linear2_entity.eval()
    model.transformer.decoder.layers[0].norm3_entity.eval()

    model.transformer.decoder.layers[1].self_attn_entity.eval()
    model.transformer.decoder.layers[1].cross_attn_entity.eval()
    model.transformer.decoder.layers[1].norm1_entity.eval()
    model.transformer.decoder.layers[1].norm2_entity.eval()
    model.transformer.decoder.layers[1].linear1_entity.eval()
    model.transformer.decoder.layers[1].linear2_entity.eval()
    model.transformer.decoder.layers[1].norm3_entity.eval()

    model.transformer.decoder.layers[2].self_attn_entity.eval()
    model.transformer.decoder.layers[2].cross_attn_entity.eval()
    model.transformer.decoder.layers[2].norm1_entity.eval()
    model.transformer.decoder.layers[2].norm2_entity.eval()
    model.transformer.decoder.layers[2].linear1_entity.eval()
    model.transformer.decoder.layers[2].linear2_entity.eval()
    model.transformer.decoder.layers[2].norm3_entity.eval()

    model.transformer.decoder.layers[3].self_attn_entity.eval()
    model.transformer.decoder.layers[3].cross_attn_entity.eval()
    model.transformer.decoder.layers[3].norm1_entity.eval()
    model.transformer.decoder.layers[3].norm2_entity.eval()
    model.transformer.decoder.layers[3].linear1_entity.eval()
    model.transformer.decoder.layers[3].linear2_entity.eval()
    model.transformer.decoder.layers[3].norm3_entity.eval()

    model.entity_class_embed.eval()
    model.entity_bbox_embed.eval()


    return model

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
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model = initialize_and_freeze(model, first_part_checkpoint_path='/p/scratch/hai_1008/kromm3/RelTR/ckpt/run_first_part/checkpoint0248_.pth')
    model.to(device)

    writer = create_writer(log_dir='/p/project/hai_1008/kromm3/RelTR/logs/run_full_sim_1')

    model_without_ddp = model
    if args.distributed:
         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
         model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    #dataset_train = build_dataset(image_set='train', args=args)
    #dataset_val = build_dataset(image_set='val', args=args)

    dataset_train = build_carla_dataset(args,
    anno_file="/p/home/jusers/kromm3/juwels/master/RelTR/datasets/annotations/Carla/train_dataset_pre.json",
    transform=make_transforms('train')  # Or custom transform pipeline
    )

    dataset_val = build_carla_dataset(args,
    anno_file="/p/home/jusers/kromm3/juwels/master/RelTR/datasets/annotations/Carla/test_dataset_pre.json",
    transform=make_transforms('val')  # Or custom transform pipeline
    )

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
         
        #if args.distributed:
        print0(f"Resume training with checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')

        #state_dict = torch.load("checkpoint.pth")
        from collections import OrderedDict
        state_dict = checkpoint['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = "module." + k  # Add 'module.' prefix
            print(new_key)
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=True)

        #model.load_state_dict(checkpoint['model'], strict=True)
        # else:
        #     print(f"Resume training with checkpoint {args.resume}")
        #     checkpoint = torch.load(args.resume, map_location='cpu')
        #     new_state_dict = OrderedDict()
        #     state_dict = checkpoint['model']
        #     for k, v in state_dict.items():
        #         name = k[7:] if k.startswith("module.") else k # remove `module.`
        #         new_state_dict[name] = v
        #     model.load_state_dict(new_state_dict, strict=True)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth'] # anti-crash
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)

        add_all_stats(writer, train_stats, test_stats, epoch)

        # for evaluation logs
        # if coco_evaluator is not None:
        #     (output_dir / 'eval').mkdir(exist_ok=True)
        #     if "bbox" in coco_evaluator.coco_eval:
        #         filenames = ['latest.pth']
        #         if epoch % 50 == 0:
        #             filenames.append(f'{epoch:03}.pth')
        #         for name in filenames:
        #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                         output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
