from __future__ import print_function

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os
import argparse
import torch
from datasets import get_augumentation, CocoDataset
from utils import EFFICIENTDET
from models import EfficientDet
from tqdm import tqdm


def evaluate_coco(dataset, model, device, eff_size, threshold=0.05):
    print('is_training: ', model.is_training)
    model.eval()
    with torch.no_grad():
        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            img_shape = dataset.load_image(index).shape
            data = dataset[index]
            images = data['image'].unsqueeze(0).to(device)
            # run network
            scores, labels, boxes = model(images)
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            print('scores: ', scores.size())
            print('boxes: ', boxes.size())
            # correct boxes for image scale
            boxes[:, 0] = boxes[:, 0]*img_shape[1] / eff_size
            boxes[:, 1] = boxes[:, 1]*img_shape[0] / eff_size
            boxes[:, 2] = boxes[:, 2]*img_shape[1] / eff_size
            boxes[:, 3] = boxes[:, 3]*img_shape[0] / eff_size

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]
                    if score < threshold:
                        break
                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }
                    results.append(image_result)
            image_ids.append(dataset.image_ids[index])

        if not len(results):
            print('Not result in Evaluation')
            return 0

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(
            dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(
            '{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EfficientDet Training With Pytorch')
    parser.add_argument('--dataset_root', default='/root/data/VOCdevkit/',
                        help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/]')
    parser.add_argument('--weight', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = []
    if(args.weight is not None):
        resume_path = str(args.weight)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(
            args.weight, map_location=lambda storage, loc: storage)
        num_class = checkpoint['num_class']
        network = checkpoint['network']
    dataset = CocoDataset(root_dir=args.dataset_root, set_name='val2017', transform=get_augumentation(
        phase='test', width=EFFICIENTDET[network]['input_size'], height=EFFICIENTDET[network]['input_size']))

    model = EfficientDet(num_classes=num_class,
                         network=network,
                         W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                         D_class=EFFICIENTDET[network]['D_class'],
                         is_training=False,
                         threshold=0.05,
                         iou_threshold=0.5
                         )
    
    if(args.weight is not None):
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
    model = model.to(device)
    evaluate_coco(dataset=dataset, model=model, eff_size=EFFICIENTDET[network]['input_size'],
                  device=device, threshold=0.05)
