import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.cm as cm


#VehicleTurningRight_7_00008900_17267

# === Klassen ===
CLASSES = [ 
    'ground', 'road', 'side walk', 'bridge', 'pole', 'traffic light',
    'traffic sign', 'person', 'car', 'truck', 'bicycle'
]
REL_CLASSES = [
    'on', 'attached to', 'on right side of', 'parking on', 'on left side of',
    'same road line as', 'on right lane of', 'on opposing side of', 'on left lane of',
    'driving from right to left', 'driving from left to right', 'on middle lane of',
    'infront of', 'behind', 'riding', 'next to', 'turning right on', 'driving on',
    'turning left on', 'is hitting'
]

# === Deine Annotation ===
annotation = {
    "labels": [7, 10, 2, 8, 1, 5, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4],
    "boxes": [[1193, 346, 1243, 435], [1165, 379, 1258, 437], [0, 299, 1919, 1079], [753, 500, 1138, 878], [0, 313, 1918, 1079], [816, 218, 821, 232], [729, 179, 935, 328], [867, 217, 872, 232], [926, 216, 931, 232], [928, 215, 1064, 311], [1109, 163, 1127, 328], [1199, 149, 1231, 327], [942, 103, 1103, 338], [592, 57, 791, 351], [1113, 0, 1180, 375], [182, 0, 294, 414]],
    "rel_annotations": [[0, 1, 14], [1, 2, 0], [3, 4, 2], [3, 4, 6], [5, 6, 1], [7, 6, 1], [8, 6, 1], [9, 2, 0], [6, 2, 0], [10, 2, 0], [11, 2, 0], [12, 2, 0], [13, 2, 0], [14, 2, 0], [15, 2, 0]]
}

#image = Image.open(image_path)

# === Hilfsfunktionen ===
def round_box(box, precision=2):
    return tuple(round(float(v), precision) for v in box)

def show_annotated_graph(image_path, annotation):
    image = Image.open(image_path)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 10))
    ax_left.imshow(image)
    ax_left.axis('off')

    cmap = cm.get_cmap('tab20')
    color_idx = 0
    entity_color_map = {}

    # Box-Namen und Gruppierung
    subj_boxes = [annotation["boxes"][s] for s, _, _ in annotation["rel_annotations"]]
    obj_boxes = [annotation["boxes"][o] for _, o, _ in annotation["rel_annotations"]]
    keep_queries = list(range(len(annotation["rel_annotations"])))

    sub_bboxes_scaled = subj_boxes
    obj_bboxes_scaled = obj_boxes
    indices = list(range(len(sub_bboxes_scaled)))

    probas_sub = [annotation["labels"][s] for s, _, _ in annotation["rel_annotations"]]
    probas_obj = [annotation["labels"][o] for _, o, _ in annotation["rel_annotations"]]
    probas_rel = [r for _, _, r in annotation["rel_annotations"]]

    unique_subj_boxes = {}
    unique_obj_boxes = {}
    subj_box_to_node = {}
    obj_box_to_node = {}
    duplicate_subj_indices = defaultdict(list)
    duplicate_obj_indices = defaultdict(list)

    for idx, sub_box in zip(keep_queries, sub_bboxes_scaled):
        box_key = round_box(sub_box)
        if box_key not in unique_subj_boxes:
            subj_cls = CLASSES[probas_sub[idx]]
            node_name = f"Subj-{subj_cls}-{len(unique_subj_boxes)}"
            unique_subj_boxes[box_key] = node_name
        subj_box_to_node[idx] = unique_subj_boxes[box_key]
        duplicate_subj_indices[unique_subj_boxes[box_key]].append(idx)

    for idx, obj_box in zip(keep_queries, obj_bboxes_scaled):
        box_key = round_box(obj_box)
        if box_key not in unique_obj_boxes:
            obj_cls = CLASSES[probas_obj[idx]]
            node_name = f"Obj-{obj_cls}-{len(unique_obj_boxes)}"
            unique_obj_boxes[box_key] = node_name
        obj_box_to_node[idx] = unique_obj_boxes[box_key]
        duplicate_obj_indices[unique_obj_boxes[box_key]].append(idx)

    # === Bounding Boxes zeichnen ===
    for idx in keep_queries:
        subj_node = subj_box_to_node[idx]
        obj_node = obj_box_to_node[idx]

        subj_color = entity_color_map.setdefault(subj_node, cmap(color_idx)); color_idx += 1
        obj_color = entity_color_map.setdefault(obj_node, cmap(color_idx)); color_idx += 1

        subj_box_key = [k for k, v in unique_subj_boxes.items() if v == subj_node][0]
        obj_box_key = [k for k, v in unique_obj_boxes.items() if v == obj_node][0]

        if idx == duplicate_subj_indices[subj_node][0]:
            sxmin, symin, sxmax, symax = subj_box_key
            subj_cls = subj_node.split('-')[1]
            rect_s = patches.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                       fill=False, edgecolor=subj_color, linewidth=3)
            ax_left.add_patch(rect_s)
            ax_left.text(sxmin, symin - 5, subj_cls, fontsize=10, color=subj_color,
                         verticalalignment='bottom', weight='bold',
                         bbox=dict(facecolor='white', edgecolor=subj_color,
                                   boxstyle='round,pad=0.3', alpha=0.7))

        if idx == duplicate_obj_indices[obj_node][0]:
            oxmin, oymin, oxmax, oymax = obj_box_key
            obj_cls = obj_node.split('-')[1]
            rect_o = patches.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                       fill=False, edgecolor=obj_color, linewidth=3)
            ax_left.add_patch(rect_o)
            ax_left.text(oxmin, oymin - 5, obj_cls, fontsize=10, color=obj_color,
                         verticalalignment='bottom', weight='bold',
                         bbox=dict(facecolor='white', edgecolor=obj_color,
                                   boxstyle='round,pad=0.3', alpha=0.7))

    # === Beziehungsgraph ===
    edge_relation_map = defaultdict(list)
    for idx in keep_queries:
        subj_node = subj_box_to_node[idx]
        obj_node = obj_box_to_node[idx]
        rel_cls = REL_CLASSES[probas_rel[idx]]
        edge_relation_map[(subj_node, obj_node)].append(rel_cls)

    G = nx.DiGraph()
    for (subj_node, obj_node), rels in edge_relation_map.items():
        G.add_node(subj_node)
        G.add_node(obj_node)
        combined_rel = " / ".join(sorted(set(rels)))
        G.add_edge(subj_node, obj_node, label=combined_rel)

    node_colors = [entity_color_map[n] for n in G.nodes()]
    pos = nx.spring_layout(G, k=1.2, iterations=50)

    nx.draw(G, pos, ax=ax_right, with_labels=True,
            node_color=node_colors, node_size=2000,
            font_size=10, font_weight='bold', arrowsize=20,
            edgecolors='black')

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_color='blue', font_size=10)

    ax_right.set_title('Relationships Graph', fontsize=16)
    ax_right.axis('off')

    plt.tight_layout()
    plt.show()

    # === Aufruf mit Bildpfad ===

# === 🔁 Lade DEIN Originalbild ===
# Ersetze den Pfad durch deinen lokalen Dateipfad
image_path = "F:\\scenario_runner-0.9.15\\Data\\_out\\VehicleTurningRight_7\\rgb\\filtered\\00008900.png"
show_annotated_graph(image_path, annotation)