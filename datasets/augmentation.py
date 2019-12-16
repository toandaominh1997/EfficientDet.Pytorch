import albumentations as albu
from albumentations.pytorch.transforms import ToTensor
import torch
import numpy as np
import cv2

import skimage.io
import skimage.transform
import skimage.color
import skimage
from .encoder import DataEncoder

def get_augumentation(phase, width=512, height=512, min_area=0., min_visibility=0.):
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.LongestMaxSize(max_size=height, always_apply=True),
            albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True),
            albu.augmentations.transforms.RandomResizedCrop(
                                 height=height,
                                 width=width, p=0.5),
            albu.augmentations.transforms.Flip(),
            albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.5,
                                              contrast_limit=0.4),
                albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=20, b_shift_limit=15,
                              g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=5,
                                        sat_shift_limit=5),
                albu.NoOp()
            ]),
            albu.CLAHE(p=0.8),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    list_transforms.extend([
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensor()
    ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(list_transforms, bbox_params=albu.BboxParams(format='pascal_voc', min_area=min_area,
                                                                     min_visibility=min_visibility, label_fields=['category_id']))


def detection_collate(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]

    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5))*-1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot
                annot_padded[idx, :len(annot), 4] = lab
    return (torch.stack(imgs, 0), torch.FloatTensor(annot_padded))

encoder = DataEncoder()
def collate_fn(batch):
    '''Pad images and encode targets.
    As for images are of different sizes, we need to pad them to the same size.
    Args:
        batch: (list) of images, cls_targets, loc_targets.
    Returns:
        padded images, stacked cls_targets, stacked loc_targets.
    '''
    imgs = [x['image'] for x in batch]
    boxes = [x['bboxes'] for x in batch]
    labels = [x['category_id'] for x in batch]

    h = w = 512
    num_imgs = len(imgs)
    inputs = torch.zeros(num_imgs, 3, h, w)

    loc_targets = []
    cls_targets = []
    print('boxes: ', boxes)
    for i in range(num_imgs):
        inputs[i] = imgs[i]
        loc_target, cls_target = encoder.encode(boxes[i], labels[i], input_size=(w,h))
        loc_targets.append(loc_target)
        cls_targets.append(cls_target)
    return inputs, torch.stack(loc_targets), torch.stack(cls_targets)