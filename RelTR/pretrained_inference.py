# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from models import custom_build_model, build_model
from visualization_techniques import visualize_attention_map, visualize_multiple_attention_maps
import matplotlib.patches as patches

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    # parser.add_argument('--img_path', type=str, default='/p/scratch/hai_1008/kromm3/CityScapes/leftImg8bit/test/munich/munich_000195_000019_leftImg8bit.png',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train_extra\\bayreuth\\bayreuth_000000_000003_leftImg8bit.png',
    #                    help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='F:\\scenario_runner-0.9.15\\Data\\_out\\OppositeVehicleRunningRedLight_3_3\\rgb\\filtered\\00001758.png',
                        #help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\test\\munich\\munich_000008_000019_leftImg8bit.png',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='F:\\scenario_runner-0.9.15\\Data\\_out\\OppositeVehicleRunningRedLight_5_3\\rgb\\filtered\\00002064.png',
    #                     help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\cologne\\cologne_000082_000019_leftImg8bit.png',
    #                      help="Path of the test image")
    parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\test\\munich\\munich_000230_000019_leftImg8bit.png',
                          help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\aachen\\aachen_000157_000019_leftImg8bit.png',
    #                      help="Path of the test image")
    #parser.add_argument('--img_path', type=str, default='S:\\Datasets\\CityScapes\\leftImg8bit\\train\\zurich\\zurich_000061_000019_leftImg8bit.png',
    #                      help="Path of the test image")  

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


    label_id = {
    'n': 0 , 'ground' : 1, 'road' : 2, 'side walk' : 3, 'bridge' : 4, 'pole' : 5, 'traffic light' : 6, 'traffic sign' : 7, 'person' : 8, 'car' : 9, 'truck' : 10, 'bicycle' : 11
    }

    
    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume, weights_only=False, map_location='cpu')
    state_dict = ckpt['model']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    print(f"Model loaded successfully from {args.resume}")

    # Load image
    img_path = args.img_path
    im = Image.open(img_path).convert('RGB')

    # Transform
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(im).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(img)

    # Prediction probabilities
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    scores = probas.max(-1).values

    # Confidence threshold
    conf_thresh = 0.99
    keep = scores > conf_thresh

    # Rescale boxes
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    classes = probas[keep].argmax(-1)

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(im)

    # Distinct colors
    num_colors = len(classes)
    color_map = plt.cm.get_cmap('hsv', num_colors + 1)

    for i, (box, cls_idx) in enumerate(zip(bboxes_scaled, classes)):
        xmin, ymin, xmax, ymax = box.tolist()
        color = color_map(i)  # RGB tuple

        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Lookup class name
        class_name = "unknown"
        for name, idx in label_id.items():
            if idx == cls_idx.item():
                class_name = name
                break

        ax.text(
        xmin, ymin - 5, f'{class_name}',
        fontsize=10, color='white',
        bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3', linewidth=1)
    )


    ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)