from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT

from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
from .config import *
import torch
import cv2
import numpy as np
import torch 

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    imgs = [s['img'] for s in batch]
    annots = [s['annot'] for s in batch]
    scales = [s['scale'] for s in batch]
    max_num_annots = max(annot.shape[0] for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5))*-1
    if max_num_annots > 0:
        # annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    return {'img': torch.stack(imgs, 0).permute(0, 3, 1, 2), 'annot': torch.tensor(annot_padded).float(), 'scale': scales}


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
