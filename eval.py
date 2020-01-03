import os
import numpy as np 
import argparse
import torch
from tqdm import tqdm

from datasets import VOCDetection, COCODetection, get_augumentation, detection_collate

from torch.utils.data import DataLoader
from models.efficientdet import EfficientDet
from utils import EFFICIENTDET


parser = argparse.ArgumentParser(
    description='EfficientDet Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='/root/data/VOCdevkit/',
                    help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/]')
parser.add_argument('--network', default='efficientdet-d0',
                    help='Choose model for training')
parser.add_argument('-t', '--threshold', default=0.4,
                    type=float, help='Visualization threshold')
parser.add_argument('-it', '--iou_threshold', default=0.5,
                    type=float, help='Visualization threshold')
parser.add_argument('--weights', default='./weights/checkpoint_VOC_efficientdet-d1_22.pth', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_worker', default=8, type=int,
                    help='Number of workers used in dataloading')

parser.add_argument('--device', default=[0], type=list,
                    help='Use CUDA to train model')
args = parser.parse_args()



if(args.weights is not None):
    resume_path = str(args.weights)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(
        args.weights, map_location=lambda storage, loc: storage)
    args.num_class = checkpoint['num_class']
    args.network = checkpoint['network']
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

if(args.dataset == 'VOC'):
    valid_dataset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'test')],
                                 transform=get_augumentation(phase='valid', width=EFFICIENTDET[args.network]['input_size'], height=EFFICIENTDET[args.network]['input_size']))
elif(args.dataset == 'COCO'):
    valid_dataset = COCODetection(root=args.dataset_root,
                                  transform=get_augumentation(phase='valid', width=EFFICIENTDET[args.network]['input_size'], height=EFFICIENTDET[args.network]['input_size']))

valid_dataloader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=args.num_worker,
                              shuffle=False,
                              collate_fn=detection_collate,
                              pin_memory=False)


model = model.cuda()

def val_coco(threshold=0.5):
    model.eval()
    with torch.no_grad():
        results = []
        image_ids = []
        for idx, (images, annotations) in enumerate(valid_dataloader):
            images = images.to(device)
            annotations = annotations.to(device)
            scores, labels, boxes = model(images)
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()
            if(boxes.shape[0] > 0):
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[: 1]
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    if score < threshold:
                        break 
        #             image_result = {
        #                 'image_id': ,
        #                 'category_id': ,
        #                 'score': float(score),
        #                 'bbox': box.tolist(),
        #             }
        #             results.append(image_result)
        # if(len(results)==0):
        #     return None
        # json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)
        # # load results in COCO evaluation tool
        # coco_true = dataset.coco
        # coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # # run COCO evaluation
        # coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        # coco_eval.params.imgIds = image_ids
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
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

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

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
def eval_voc(iou_threshold=0.5):
    
    
    all_detections = [[None for i in range(valid_dataset.__num_class__())] for j in range(len(valid_dataset))]
    all_annotations = [[None for i in range(valid_dataset.__num_class__())] for j in range(len(valid_dataset))]
    model.eval()
    for idx, (images, annotations) in enumerate(tqdm(valid_dataloader)):
        images = images.cuda()
        annotations = annotations.cuda()
        with torch.no_grad():
            scores, classification, transformed_anchors = model(images)
            if(scores.shape[0]>0):
                pred_annots = []
                for j in range(scores.shape[0]):
                    bbox = transformed_anchors[[j], :][0]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    idx_name = int(classification[[j]])
                    score = scores[[j]].cpu().numpy()
                    pred_annots.append([x1, y1, x2, y2, score, idx_name])    
                pred_annots = np.vstack(pred_annots)
                for label in range(valid_dataset.__num_class__()):
                    all_detections[idx][label] = pred_annots[pred_annots[:, -1] == label, :-1]
            else:
                for label in range(valid_dataset.__num_class__()):
                    all_detections[idx][label] = np.zeros((0, 5))
                
            annotations = annotations[0].cpu().numpy()
            for label in range(valid_dataset.__num_class__()):
                all_annotations[idx][label] = annotations[annotations[:, 4] == label, :4].copy()
    print('\t Start caculator mAP ...')
    average_precisions = {}

    for label in range(valid_dataset.__num_class__()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(valid_dataset.__num_class__()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    print('\tmAP:')
    mAPS = []
    for label in range(valid_dataset.__num_class__()):
        label_name = valid_dataset.label_to_name(label)
        mAPS.append(average_precisions[label][0])
        print('{}: {}'.format(label_name, average_precisions[label][0]))
    print('total mAP: {}'.format(np.mean(mAPS)))
    return average_precisions



                



if __name__ == '__main__':
    eval_voc()