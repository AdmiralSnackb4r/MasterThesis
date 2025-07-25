import argparse
import torch
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import util.misc as utils
import datetime
import random
from torch.utils.data import DataLoader, DistributedSampler
from models import build_model, custom_build_model
#import torchvision.transforms as T
from torchvision.transforms import ToPILImage
from datasets import build_custom_dataset, get_coco_api_from_dataset, build_merged_dataset
from PIL import Image
from pre_engine import train_one_epoch, evaluate
import torchvision.transforms.v2 as v2
from distributed_utils import *
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import datasets.transforms as T

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    print0("rescale", out_bbox.shape)
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def get_args_parser():

    # Training parameters
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=400, type=int)
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
                        help="Number of attention heads inside the transformer's attentions") #8
    parser.add_argument('--num_entities', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # run_6 enc/dec layers = 4, nheads = 4, --> higher loses

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
    parser.add_argument('--dataset', default='pretrain')
    parser.add_argument('--ann_path', default='./data/vg/', type=str)
    #parser.add_argument('--datapath', default='S:\\Datasets\\CityScapes\\leftImg8bit', type=str)
    parser.add_argument('--datapath', default="/p/scratch/hai_1008/kromm3", help='path to data')
    parser.add_argument('--on_cluster', type=bool, default=True)

    parser.add_argument('--output_dir', default='/p/scratch/hai_1008/kromm3/RelTR/ckpt/run_first_part_with_sim',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/p/scratch/hai_1008/kromm3/RelTR/ckpt/run_first_part_with_sim/checkpoint_bup_0419_.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=42, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

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
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
    

def main(args):

    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    writer = create_writer(log_dir='/p/project/hai_1008/kromm3/RelTR/logs/run_first_part_with_sim')

    model, criterion, postprocessors = custom_build_model(args)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.distributed:
        print0('number of params:', n_parameters)
    else:
        print('number of params:', n_parameters)

    param_dicts = [
         {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
         {
             "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": args.lr_backbone,
         },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.on_cluster:
        dataset_train = build_merged_dataset(args, anno_file="/p/home/jusers/kromm3/juwels/master/RelTR/datasets/annotations/Merged/merged_with_carla_train.json", transform=make_transforms('train'))
        dataset_val = build_merged_dataset(args=args, anno_file='/p/home/jusers/kromm3/juwels/master/RelTR/datasets/annotations/Merged/merged_valid.json', transform=make_transforms('val'))
    else:
        dataset_train = build_custom_dataset(args=args, anno_file='datasets\\annotations\\train_dataset.json', transform=make_transforms('train'))
        dataset_val = build_custom_dataset(args=args, anno_file='datasets\\annotations\\valid_dataset.json', transform=make_transforms('val'))


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
            new_key = k
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


    if args.eval:
        if args.distributed:
            print0('It is the {}th checkpoint'.format(checkpoint['epoch']))
        else:
            print('It is the {}th checkpoint'.format(checkpoint['epoch']))

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    if args.distributed:
        print0("Start training")
    else:
        print("Start training")

    best_val_loss = float('inf')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        eval_stats = evaluate(model, criterion, data_loader_val, device)

        add_stats(writer, train_stats, eval_stats, epoch)

        eval_loss = sum(eval_stats['loss']) / len(eval_stats['loss'])
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_anti_crash.pth'] # anti-crash
            if eval_loss < best_val_loss:
               best_val_loss = eval_loss
               checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_.pth')
            if (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint_bup_{epoch:04}_.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    close_writer(writer)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.distributed:
        print0('Training time {}'.format(total_time_str))
    else:
        print('Training time {}'.format(total_time_str))




if __name__ == "__main__":
    parser = argparse.ArgumentParser('RelTR pre training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)