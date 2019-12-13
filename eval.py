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

parser = argparse.ArgumentParser(
    description='EfficientDet Training With Pytorch')
parser.add_argument('--dataset_root', default='/root/data/VOCdevkit/',
                    help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/]')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
args = parser.parse_args()


def evaluate_coco(dataset, model, threshold=0.05):

    model.eval()

    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in tqdm(range(len(dataset))):
            data = dataset[index]
            scale = 1.0
            images = data['image'].permute(2, 0, 1).unsqueeze(0).cuda()
            print('images: ', images.size())
            # run network
            scores, labels, boxes = model(data['image'].permute(
                2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

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


checkpoint = []
if(args.resume is not None):
    resume_path = str(args.resume)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(
        args.resume, map_location=lambda storage, loc: storage)
    num_class = checkpoint['num_class']
    network = checkpoint['network']
dataset = CocoDataset(root_dir=args.dataset_root, set_name='train2017', transform=get_augumentation(
    phase='test', width=EFFICIENTDET[network]['input_size'], height=EFFICIENTDET[network]['input_size']))
model = EfficientDet(num_classes=num_class,
                     network=network,
                     W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                     D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                     D_class=EFFICIENTDET[network]['D_class'],
                     is_training=False
                     )

if __name__ == '__main__':
    evaluate_coco(dataset, model)
