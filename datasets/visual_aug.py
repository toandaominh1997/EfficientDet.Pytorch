import cv2
from augmentation import get_augumentation
from voc0712 import VOCDetection
import matplotlib.pyplot as plt
EFFICIENTDET = {
    'efficientdet-d0': {'input_size': 512,
                        'backbone': 'B0',
                        'W_bifpn': 64,
                        'D_bifpn': 2,
                        'D_class': 3},
    'efficientdet-d1': {'input_size': 640,
                        'backbone': 'B1',
                        'W_bifpn': 88,
                        'D_bifpn': 3,
                        'D_class': 3},
    'efficientdet-d2': {'input_size': 768,
                        'backbone': 'B2',
                        'W_bifpn': 112,
                        'D_bifpn': 4,
                        'D_class': 3},
}


# Functions to visualize bounding boxes and class labels on an image.
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)
    # class_name = class_idx_to_name[class_id]
    # ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    # cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    # cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(
            img, bbox, annotations['category_id'][idx], category_id_to_name)
    # plt.figure(figsize=(12, 12))
    # plt.imshow(img)
    return img


dataset_root = '/root/data/VOCdevkit'
network = 'efficientdet-d0'
dataset = VOCDetection(root=dataset_root,
                       transform=get_augumentation(phase='train', width=EFFICIENTDET[network]['input_size'], height=EFFICIENTDET[network]['input_size']))


def visual_data(data, name):
    img = data['image']
    bboxes = data['bboxes']
    annotations = {'image': data['image'], 'bboxes': data['bboxes'], 'category_id': range(
        len(data['bboxes']))}
    category_id_to_name = {v: v for v in range(len(data['bboxes']))}

    img = visualize(annotations, category_id_to_name)
    cv2.imwrite(name, img)


for i in range(20, 25):
    visual_data(dataset[i], "name"+str(i)+".png")
