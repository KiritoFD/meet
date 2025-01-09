import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比 (Intersection over Union, IoU)。

    参数:
        box1 (tuple): 第一个边界框，格式为 (x1, y1, x2, y2)。
        box2 (tuple): 第二个边界框，格式为 (x1, y1, x2, y2)。

    返回:
        float: IoU 值，范围在 0 到 1 之间。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    
    iou = intersection_area / union_area
    return iou