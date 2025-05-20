

import torch
import argparse
from pathlib import Path
import util.misc as utils
from datasets import build_custom_dataset, build_carla_dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import datasets.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
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
    parser.add_argument('--datapath', default='F:\\scenario_runner-0.9.15\\Data', type=str)


    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser



def main(args):

    def make_coco_transforms(image_set):

        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

        if image_set == 'train':
            return T.Compose([
                #T.RandomHorizontalFlip(), Contradict ground-truth labels
                T.SparseColorNoise(),
                T.RandomColorColumnsPadding(1, 120, (1080, 1920)),
                T.RandomGrayscale(0.3),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomAdjustSharpness(),
                        T.RandomGaussianBlur(),
                        T.RandomColorJitter(),
                        T.RandomAutocontrast(),
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

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    test = build_carla_dataset(args=args, anno_file='datasets\\annotations\\Carla\\train_dataset_pre.json', transform=make_coco_transforms('train'))
    sampler = torch.utils.data.RandomSampler(test)

    batch_sampler = torch.utils.data.BatchSampler(
        sampler, args.batch_size, drop_last=True
    )

    dataloader = DataLoader(test, batch_sampler=batch_sampler,
                            collate_fn=utils.collate_fn, num_workers=args.num_workers)
    

    batch = next(iter(dataloader))
    images, targets = batch

    # Take the first image tensor
    img_tensor = images.tensors[0]
    target = targets[0]

    # Convert tensor to PIL image for display
    img_pil = F.to_pil_image(img_tensor)

    # Plot image
    fig, ax = plt.subplots(1)
    ax.imshow(img_pil)

    # Draw bounding boxes (assumes target["boxes"] is a tensor of [xmin, ymin, xmax, ymax])
    boxes = target["boxes"]

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    plt.title("Transformed Image with Bounding Boxes")
    plt.axis('off')
    plt.show()
    
    #print(test.__getitem__(1500))
    
    # for i in range(len(test)):
    #     print(i)
    #     batch = next(iter(dataloader))
    #     #print(batch)
    #     #break

    #print(next(iter(dataloader)))
    
    #print(type(next(iter(dataloader))[1][0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RelTR test the dataloader script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)