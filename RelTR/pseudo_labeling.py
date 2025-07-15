import math
import json
import torch
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import util.misc as utils
from models import build_model
import datasets.transforms as T
from torchvision.transforms import ToPILImage
from datasets import build_merged_dataset, build_custom_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Carla classes
CLASSES = [ 'N/A', 'ground', 'road', 'side walk', 'bridge', 'pole', 'traffic light', 'traffic sign', 'person', 'car', 'truck', 'bicycle']

REL_CLASSES = [ '__background__', 'on', 'attached to', 'on right side of', 'parking on', 'on left side of', 'same road line as', 'on right lane of', 'on opposing side of', 'on left lane of', 'driving from right to left', 'driving from left to right', 'on middle lane of',
                'infront of', 'behind', 'riding', 'next to', 'turning right on', 'driving on', 'turning left on', 'is hitting']

def round_box(box, precision=2):
            return tuple(round(float(v), precision) for v in box)

def int_box(box):
     return tuple(int(v) for v in box)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)

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
    

    parser.add_argument('--datapath', default='S:\\Datasets\\CityScapes\\leftimg8bit', type=str)
    parser.add_argument('--num_workers', default=1, type=int)

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
            T.RandomResize([1333], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

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


def main(args):

    # transform = T.Compose([
    #     T.Resize(800),
    #     T.ToTensor(),
    #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    real_annos_with_rels = {}

    denorm = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
    )

    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    dataset_annotate = build_custom_dataset(args=args, anno_file='datasets\\annotations\\CityScapes\\train_dataset.json', transform=make_transforms('val'))
    sampler_annotate = torch.utils.data.SequentialSampler(dataset_annotate)

    data_loader_annotate = DataLoader(dataset_annotate, args.batch_size, sampler=sampler_annotate,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    for batch_idx, (samples, targets, img_id) in tqdm(enumerate(data_loader_annotate), total=len(data_loader_annotate), desc="Annotating"):
        outputs = model(samples)
        img_size = targets[0]['orig_size']

        #print(img_id)

        # Compute softmax probabilities for various logits
        probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]     # [num_queries, num_rel_classes]
        probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1] # [num_queries, num_sub_classes]
        probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1] # [num_queries, num_obj_classes]
        probas_entity = outputs['pred_logits'].softmax(-1)[0, :, :-1] # [num_queries, num_entity_classes]


        # Keep entity predictions with confidence > 0.85
        keep_entity = probas_entity.max(-1).values > 0.95  # boolean mask [num_queries]
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep_entity], img_size)  # filtered boxes
        probas_entity_kept = probas_entity[keep_entity]
        labels_entity = probas_entity_kept.argmax(dim=1)

        # # Select top-k confident entities
        topk = 300
        keep_queries_entity = torch.nonzero(keep_entity, as_tuple=True)[0]
        indices_entity = torch.argsort(-probas_entity_kept.max(-1)[0])[:topk]
        keep_queries_entity = keep_queries_entity[indices_entity]

        # Keep relation queries with confidence > 0.5 in all three logits
        keep = torch.logical_and(probas.max(-1).values > 0.85,
                torch.logical_and(probas_sub.max(-1).values > 0.85, probas_obj.max(-1).values > 0.85))

        # Filter labels and boxes for subjects and objects accordingly
        labels_sub = probas_sub[keep].argmax(dim=1)
        labels_obj = probas_obj[keep].argmax(dim=1)

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], img_size)
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], img_size)

        if sub_bboxes_scaled.shape[0] == 0 or obj_bboxes_scaled.shape[0] == 0:
            continue

        #Merge boxes based on labels (you need your merge_predictions updated accordingly)
        sub_bboxes_scaled = merge_predictions(
            bboxes_scaled, sub_bboxes_scaled, labels_entity, labels_sub)#

        obj_bboxes_scaled = merge_predictions(
            bboxes_scaled, obj_bboxes_scaled, labels_entity, labels_obj)
        
        topk = 10
        keep_queries = torch.nonzero(keep, as_tuple=True)[0]
        scores = probas[keep].max(-1)[0] * probas_sub[keep].max(-1)[0] * probas_obj[keep].max(-1)[0]
        indices = torch.argsort(-scores)[:topk]
        keep_queries = keep_queries[indices]

        # if batch_idx == 10:

        #     # Step 1: Create unique subject and object box maps
        #     unique_subj_boxes = {}
        #     unique_obj_boxes = {}

        #     subj_box_to_node = {}
        #     obj_box_to_node = {}

        #     subject_filter = None  # e.g., "car" to filter only "car" subjects
        #     object_filter = None   # e.g., "traffic light" to filter only "traffic light" object

        #     duplicate_subj_indices = defaultdict(list)
        #     duplicate_obj_indices = defaultdict(list)

        #     entity_color_map = {}  # maps node_name -> color
        #     fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(22, 12))

        #     to_piller = ToPILImage()
        #     img = denorm(samples.tensors[0])

        #     ax_left.imshow(to_piller(img))
        #     ax_left.axis('off')
        #     ax_left.set_title('Detected Objects', fontsize=16)

        #     for idx, sub_box in zip(keep_queries, sub_bboxes_scaled[indices]):
        #         idx = int(idx)
        #         box_key = round_box(sub_box)
        #         if box_key not in unique_subj_boxes:
        #             subj_cls = CLASSES[probas_sub[idx].argmax()]
        #             if subject_filter is not None and subj_cls != subject_filter:
        #                 continue
        #             print(sub_box, box_key)
        #             node_name = f"{subj_cls}-X:{box_key[0]}-Y:{box_key[2]}"
        #             unique_subj_boxes[box_key] = node_name
        #         subj_box_to_node[idx] = unique_subj_boxes[box_key]
        #         duplicate_subj_indices[unique_subj_boxes[box_key]].append(idx)

        #     for idx, obj_box in zip(keep_queries, obj_bboxes_scaled[indices]):
        #         idx = int(idx)
        #         box_key = round_box(obj_box)
        #         if box_key not in unique_obj_boxes:
        #             obj_cls = CLASSES[probas_obj[idx].argmax()]
        #             if object_filter is not None and obj_cls != object_filter:
        #                 continue
        #             node_name = f"{obj_cls}-X:{box_key[0]}-Y:{box_key[2]}"
        #             unique_obj_boxes[box_key] = node_name
        #         obj_box_to_node[idx] = unique_obj_boxes[box_key]
        #         duplicate_obj_indices[unique_obj_boxes[box_key]].append(idx)

        #     # Step 2: Draw subject and object bounding boxes (only first per group)
        #     for idx in keep_queries:
        #         idx = int(idx)  # 🔥 Fix tensor key issue
        #         if idx not in subj_box_to_node or idx not in obj_box_to_node:
        #             continue
        #         subj_node = subj_box_to_node[idx]
        #         obj_node = obj_box_to_node[idx]

        #         subj_color = entity_color_map.setdefault(subj_node, 'orange')
        #         obj_color = entity_color_map.setdefault(obj_node, 'blue')

        #         subj_box_key = [k for k, v in unique_subj_boxes.items() if v == subj_node][0]
        #         obj_box_key = [k for k, v in unique_obj_boxes.items() if v == obj_node][0]

        #         if idx == duplicate_subj_indices[subj_node][0]:
        #             sxmin, symin, sxmax, symax = subj_box_key
        #             subj_cls = subj_node.split('-')[0]
        #             rect_s = plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
        #                                 fill=False, edgecolor=subj_color, linewidth=3)
        #             ax_left.add_patch(rect_s)
        #             ax_left.text(sxmin, symin - 5, subj_cls, fontsize=10, color=subj_color,
        #                         verticalalignment='bottom', weight='bold',
        #                         bbox=dict(facecolor='white', edgecolor=subj_color, boxstyle='round,pad=0.3', alpha=0.7))

        #         if idx == duplicate_obj_indices[obj_node][0]:
        #             oxmin, oymin, oxmax, oymax = obj_box_key
        #             obj_cls = obj_node.split('-')[0]
        #             rect_o = plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
        #                                 fill=False, edgecolor=obj_color, linewidth=3)
        #             ax_left.add_patch(rect_o)
        #             ax_left.text(oxmin, oymin - 5, obj_cls, fontsize=10, color=obj_color,
        #                         verticalalignment='bottom', weight='bold',
        #                         bbox=dict(facecolor='white', edgecolor=obj_color, boxstyle='round,pad=0.3', alpha=0.7))

        #     # Group relationships between same subj→obj pairs
        #     edge_relation_map = defaultdict(list)

        #     for idx in keep_queries:
        #         idx = int(idx)
        #         if idx not in subj_box_to_node or idx not in obj_box_to_node:
        #             continue
        #         subj_node = subj_box_to_node[idx]
        #         obj_node = obj_box_to_node[idx]
        #         rel_cls = REL_CLASSES[probas[idx].argmax()]

        #         edge_relation_map[(subj_node, obj_node)].append(rel_cls)

        #     # Step 4: Draw graph with aggregated relationship labels
        #     G = nx.DiGraph()

        #     for (subj_node, obj_node), rels in edge_relation_map.items():
        #         print(subj_node)
        #         G.add_node(subj_node)
        #         G.add_node(obj_node)

        #         # Combine unique relationships into a single label string
        #         combined_rel = " / ".join(sorted(set(rels)))
        #         G.add_edge(subj_node, obj_node, label=combined_rel)

        #     # Get node colors
        #     node_colors = [entity_color_map[node] for node in G.nodes()]
        #     pos = nx.spring_layout(G, k=1.2, iterations=50)

        #     # Draw nodes and edges
        #     nx.draw(
        #         G, pos, ax=ax_right, with_labels=True,
        #         node_color=node_colors, node_size=2000,
        #         font_size=10, font_weight='bold', arrowsize=20,
        #         edgecolors='black'
        #     )

        #     # Draw edge labels (multiple relationships per edge)
        #     edge_labels = nx.get_edge_attributes(G, 'label')
        #     nx.draw_networkx_edge_labels(
        #         G, pos, edge_labels=edge_labels,
        #         font_color='blue', font_size=10
        #     )

        #     ax_right.set_title('Relationships Graph', fontsize=16)
        #     ax_right.axis('off')


        #     # --------------------------------------
        #     # Step 4: Final layout
        #     # --------------------------------------
        #     fig.tight_layout()
        #     plt.savefig('pretrained_inference.png', dpi=600, bbox_inches='tight')
        #     plt.show()

        #fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(22, 12))

        labels = []
        boxes = []
        rel_annotations = []

        #to_piller = ToPILImage()
        #img = denorm(samples.tensors[0])

        # ax_left.imshow(to_piller(img))
        # ax_left.axis('off')
        # ax_left.set_title('Detected Objects', fontsize=16)

        itt = 0

        for idx, sub_box, obj_box in zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            subj_cls = probas_sub[idx].argmax()
            obj_cls = probas_obj[idx].argmax()
            rel_cls = probas[idx].argmax()

            labels.append(subj_cls.item())
            labels.append(obj_cls.item())

            boxes.append(int_box(sub_box))
            boxes.append(int_box(obj_box))

            rel_annotations.append([itt, itt+1, rel_cls.item()])

            #sxmin, symin, sxmax, symax = round_box(sub_box)
            #oxmin, oymin, oxmax, oymax = round_box(obj_box)

            # rect_s = plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
            #                          fill=False, edgecolor='orange', linewidth=3)
            # rect_o = plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
            #                          fill=False, edgecolor='blue', linewidth=3)

            # ax_left.add_patch(rect_s)
            # ax_left.add_patch(rect_o)

            # ax_left.text(sxmin, symin - 5, subj_cls, fontsize=10, color='orange',
            #                      verticalalignment='bottom', weight='bold',
            #                      bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round,pad=0.3', alpha=0.7))

            # ax_left.text(oxmin, oymin - 5, obj_cls, fontsize=10, color='blue',
            #                      verticalalignment='bottom', weight='bold',
            #                      bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3', alpha=0.7))
            
            itt += 2
            
        # fig.tight_layout()
        # #plt.savefig('pretrained_inference.png', dpi=600, bbox_inches='tight')
        # plt.show()

        
        real_annos_with_rels[f"city_{img_id[0]['file_name']}"] = {
                 'gt_&_city' : img_id[0]['gt_&_city'],
                 'labels' : labels,
                 'boxes' : boxes,
                 'rel_annotations' : rel_annotations
            }


        # break
        

        # if batch_idx > 10:
        #     break

    #print(real_annos_with_rels)

    with open("output_test.json", "w") as f:
        json.dump(real_annos_with_rels, f, indent=4)  # `indent` makes it pretty-printed


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)