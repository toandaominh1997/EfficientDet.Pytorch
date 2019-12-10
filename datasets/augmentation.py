import albumentations as albu
from albumentations.pytorch.transforms import ToTensor
import torch
import numpy as np


def get_augumentation(phase, width=512, height=512, min_area=0., min_visibility=0.):
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.RandomResizedCrop(
                height=height, width=width, p=1.0),
            albu.HorizontalFlip(p=0.5),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
                albu.VerticalFlip(p=0.5),
            ], p=0.3),
            albu.ShiftScaleRotate(),
        ])
    list_transforms.extend([
        ToTensor()
    ])
    if(phase == 'test'):
        list_transforms.extend([
            albu.Resize(width, height),
        ])
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
