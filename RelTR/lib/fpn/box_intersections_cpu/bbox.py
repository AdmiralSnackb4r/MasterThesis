import torch

def bbox_overlaps(boxes: torch.Tensor, query_boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes IoU (Intersection over Union) between `boxes` and `query_boxes`.
    
    Args:
        boxes (Tensor): Shape (N, 4), each row is (x_min, y_min, x_max, y_max)
        query_boxes (Tensor): Shape (K, 4), each row is (x_min, y_min, x_max, y_max)
    
    Returns:
        Tensor: Shape (N, K) containing IoU values.
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    boxes = torch.Tensor(boxes)
    query_boxes = torch.Tensor(query_boxes)

    # Compute intersection
    max_xy = torch.min(boxes[:, None, 2:], query_boxes[:, 2:])  # (N, K, 2)
    min_xy = torch.max(boxes[:, None, :2], query_boxes[:, :2])  # (N, K, 2)
    inter_wh = (max_xy - min_xy).clamp(min=0)  # Width and height of intersection (N, K, 2)

    intersection = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # (N, K)

    # Compute areas
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])  # (K,)

    # Compute union
    union = box_areas[:, None] + query_areas - intersection  # (N, K)

    # Compute IoU
    iou = intersection / union
    return iou


def bbox_intersections(boxes: torch.Tensor, query_boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the intersection ratio (intersection area / query box area).
    
    Args:
        boxes (Tensor): Shape (N, 4), each row is (x_min, y_min, x_max, y_max)
        query_boxes (Tensor): Shape (K, 4), each row is (x_min, y_min, x_max, y_max)
    
    Returns:
        Tensor: Shape (N, K) containing intersection ratios.
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    # Compute intersection
    max_xy = torch.min(boxes[:, None, 2:], query_boxes[:, 2:])  # (N, K, 2)
    min_xy = torch.max(boxes[:, None, :2], query_boxes[:, :2])  # (N, K, 2)
    inter_wh = (max_xy - min_xy).clamp(min=0)  # (N, K, 2)

    intersection = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # (N, K)

    # Compute query box areas
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])  # (K,)

    # Compute intersection ratio
    intersection_ratio = intersection / query_areas
    return intersection_ratio
