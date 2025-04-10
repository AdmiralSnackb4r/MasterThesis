# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from models import custom_build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='/p/scratch/hai_1008/kromm3/CityScapes/leftImg8bit/test/munich/munich_000195_000019_leftImg8bit.png',
                        help="Path of the test image")

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

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='/p/home/jusers/kromm3/jureca/master/scratch/RelTR/ckpt/run_1/checkpoint0014_.pth', help='resume from checkpoint')
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
    "road" : 0,
    "side walk" : 1,
    "parking" : 2,
    "rail track" : 3,
    "building" : 4,
    "wall" : 5,
    "fence" : 6,
    "guard rail" : 7,
    "bridge" : 8,
    "tunnel" : 9,
    "pole" : 10,
    "traffic light" : 11,
    "traffic sign" : 12,
    "vegetation" : 13,
    "terrain" : 14,
    "sky" : 15,
    "person" : 16,
    "rider" : 17,
    "car" : 18,
    "truck" : 19,
    "bus" : 20,
    "caravan" : 21,
    "trailer" : 22,
    "train" : 23,
    "motorcycle" : 24,
    "bicycle" : 25,
    "ground" : 26
    }

    
    model, _, _ = custom_build_model(args)
    ckpt = torch.load(args.resume, map_location='cpu')
    state_dict = ckpt['model']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    print(f"Model loaded successfully from {args.resume}")

    img_path = args.img_path
    im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)
    print(f"Inference done successfully")

    # keep only predictions with 0.+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = torch.tensor(probas.max(-1).values > 0.85)
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    print(bboxes_scaled)
    #topk = 10
    #keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    #indices = torch.argsort(-probas[keep_queries].max(-1)[0])[:topk]
    #keep_queries = keep_queries[indices]

    # use lists to store the outputs via up-values
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

    hooks = [
         model.backbone[-2].register_forward_hook(
             lambda self, input, output: conv_features.append(output)
         ),
    #     model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
    #         lambda self, input, output: dec_attn_weights_sub.append(output[1])
    #     ),
    #     model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
    #         lambda self, input, output: dec_attn_weights_obj.append(output[1])
    #     )
    ]
    with torch.no_grad():
        # propagate through the model
        outputs = model(img)
        #print(outputs)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        # dec_attn_weights_sub = dec_attn_weights_sub[0]
        # dec_attn_weights_obj = dec_attn_weights_obj[0]

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]
        im_w, im_h = im.size

        ##fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Single subplot
        #ax.imshow(im)

        #for i, (bxmin, bymin, bxmax, bymax) in enumerate(bboxes_scaled):
        #     class_index = probas[keep][i].argmax()
        #     for key, value in label_id.items(): # get the class name from the class index
        #         if value == class_index:
        #             class_name = key


        #     rect = plt.Rectangle((bxmin, bymin), bxmax - bxmin, bymax - bymin,
        #                         fill=False, color='blue', linewidth=2.5)
        #     ax.add_patch(rect)
        #     # Add text label for class
        #     ax.text(bxmin, bymin, class_name, fontsize=10, color='red') # add the class name to the bounding box

        # plt.tight_layout()
        # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
