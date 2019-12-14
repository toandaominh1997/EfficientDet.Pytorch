import os
import numpy as np
import argparse
import torch
from tqdm import tqdm

from datasets import VOCDetection, get_augumentation
from utils import EFFICIENTDET
from models import EfficientDet
from tqdm import tqdm


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


def eval_voc(dataset, model, device, eff_size, threshold=0.05, iou_threshold=0.5):
    all_detections = [[None for i in range(
        dataset.__num_class__())] for j in range(len(dataset))]
    all_annotations = [[None for i in range(
        dataset.__num_class__())] for j in range(len(dataset))]
    model.eval()
    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            img_shape = dataset.load_image(index).shape
            data = dataset[index]
            images = data['image'].unsqueeze(0).to(device)
            # run network
            scores, labels, boxes = model(images)
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()
            # correct boxes for image scale
            boxes[:, 0] = boxes[:, 0]*img_shape[1] / eff_size
            boxes[:, 1] = boxes[:, 1]*img_shape[0] / eff_size
            boxes[:, 2] = boxes[:, 2]*img_shape[1] / eff_size
            boxes[:, 3] = boxes[:, 3]*img_shape[0] / eff_size

            if boxes.shape[0] > 0:
                pred_annots = []
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]
                    if score < threshold:
                        break
                    pred_annots.append(
                        [box[0], box[1], box[2], box[3], float(score), label])
                pred_annots = np.vstack(pred_annots)
                for label in range(dataset.__num_class__()):
                    all_detections[index][label] = pred_annots[pred_annots[:, -1] == label, :-1]
            else:
                for label in range(dataset.__num_class__()):
                    all_detections[index][label] = np.zeros((0, 5))

            annotations = np.zeros(
                (len(data['bboxes']), 5), dtype=np.float)
            annotations[:, :4] = data['bboxes']
            annotations[:, 4] = data['category_id']
            for label in range(dataset.__num_class__()):
                all_annotations[index][label] = annotations[annotations[:, 4] == label, :4].copy(
                )
        print('\t Start caculator mAP ...')
        average_precisions = {}

        for label in range(dataset.__num_class__()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(dataset.__num_class__()):
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

        print('\tmAP:')
        mAPS = []
        for label in range(dataset.__num_class__()):
            label_name = dataset.label_to_name(label)
            mAPS.append(average_precisions[label][0])
            print('{}: {}'.format(label_name, average_precisions[label][0]))
        print('total mAP: {}'.format(np.mean(mAPS)))
        return average_precisions


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
    dataset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'val')],
                           transform=get_augumentation(phase='test', width=EFFICIENTDET[network]['input_size'], height=EFFICIENTDET[network]['input_size']))
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
    eval_voc(dataset=dataset, model=model, eff_size=EFFICIENTDET[network]['input_size'],
             device=device, threshold=0.05)
