import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
import json

from datasets import (Augmenter, CocoDataset, Normalizer,
                      Resizer, VOCDetection, collater, detection_collate,
                      get_augumentation)
from models.efficientdet import EfficientDet
from utils import EFFICIENTDET, get_state_dict


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(
        a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(
        a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) *
                        (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(
        dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()

    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = retinanet(data['img'].permute(
                2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(
                    image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(
        generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4]
                                                    == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    retinanet,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations

    all_detections = _get_detections(
        generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(
                    np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / \
            np.maximum(true_positives + false_positives,
                       np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    print('\nmAP:')
    avg_mAP = []
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))
        avg_mAP.append(average_precisions[label][0])
    print('avg mAP: {}'.format(np.mean(avg_mAP)))
    return np.mean(avg_mAP), average_precisions


def evaluate_coco(dataset, model, threshold=0.05):

    model.eval()

    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = model(data['img'].permute(
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

        model.train()

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EfficientDet Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default='/root/data/VOCdevkit/',
                        help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/]')
    parser.add_argument('-t', '--threshold', default=0.4,
                        type=float, help='Visualization threshold')
    parser.add_argument('-it', '--iou_threshold', default=0.5,
                        type=float, help='Visualization threshold')
    parser.add_argument('--weight', default='./checkpoint_VOC_efficientdet-d0_248.pth', type=str,
                        help='Checkpoint state_dict file to resume training from')
    args = parser.parse_args()

    if(args.weight is not None):
        resume_path = str(args.weight)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(
            args.weight, map_location=lambda storage, loc: storage)
        params = checkpoint['parser']
        args.num_class = params.num_class
        args.network = params.network
        model = EfficientDet(
            num_classes=args.num_class,
            network=args.network,
            W_bifpn=EFFICIENTDET[args.network]['W_bifpn'],
            D_bifpn=EFFICIENTDET[args.network]['D_bifpn'],
            D_class=EFFICIENTDET[args.network]['D_class'],
            is_training=False,
            threshold=args.threshold,
            iou_threshold=args.iou_threshold)
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    if(args.dataset == 'VOC'):
        valid_dataset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'test')],
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
        evaluate(valid_dataset, model)
    else:
        valid_dataset = CocoDataset(root_dir=args.dataset_root, set_name='val2017',
                                    transform=transforms.Compose([Normalizer(), Resizer()]))
        evaluate_coco(valid_dataset, model)
