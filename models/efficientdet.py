import torch
import torch.nn as nn
import math
from models.efficientnet import EfficientNet
from models.bifpn import BIFPN
from .retinahead import RetinaHead
from models.module import RegressionModel, ClassificationModel, Anchors, ClipBoxes, BBoxTransform
from torchvision.ops import nms
from .losses import FocalLoss
MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}


class EfficientDet(nn.Module):
    def __init__(self,
                 num_classes,
                 network='efficientdet-d0',
                 D_bifpn=3,
                 W_bifpn=88,
                 D_class=3,
                 is_training=True,
                 threshold=0.01,
                 iou_threshold=0.5):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNet.from_pretrained(MODEL_MAP[network])
        # self.is_training = is_training
        self.neck = BIFPN(in_channels=self.backbone.get_list_features()[-5:],
                          out_channels=W_bifpn,
                          stack=D_bifpn,
                          num_outs=5)
        self.bbox_head = RetinaHead(num_classes=num_classes,
                                    in_channels=W_bifpn)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.threshold = threshold
        self.iou_threshold = iou_threshold

        # ============== original code starts ===============
        """The following code forces all weights to be random, which does not make sense at all!"""
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        # self.freeze_bn()
        # ============== original code ends ===============

        self.criterion = FocalLoss()

    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x[-5:])
        return x

    def forward(self, inputs):
        if self.training:
            inputs, annotations = inputs
        else:
            inputs = inputs
        x = self.extract_feat(inputs)
        outs = self.bbox_head(x)
        classification = torch.cat([out for out in outs[0]], dim=1)
        regression = torch.cat([out for out in outs[1]], dim=1)
        anchors = self.anchors(inputs)
        # if anchors.dtype != inputs.dtype:  # used for mixed precision training
        #     anchors = anchors.type_as(inputs)
        if self.training:
            return self.criterion(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, inputs)
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            nms_scores = []
            nms_class = []
            anchors = []
            for idx, score in enumerate(scores):
                scores_over_thresh = (score > self.threshold)[:, 0]
                if scores_over_thresh.sum() == 0:
                    print('No boxes to NMS')
                    # no boxes to NMS, just return
                    # return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
                    continue
                cls_tmp = classification[idx, scores_over_thresh, :]
                trf_anchors = transformed_anchors[idx, scores_over_thresh, :]
                scores_tmp = scores[idx, scores_over_thresh, :]
                anchors_nms_idx = nms(trf_anchors, scores_tmp[:, 0],
                                      iou_threshold=self.iou_threshold)
                nms_scores_tmp, nms_class_tmp = cls_tmp[anchors_nms_idx, :].max(dim=1)
                nms_scores.append(nms_scores_tmp)
                nms_class.append(nms_class_tmp)
                anchors.append(trf_anchors[anchors_nms_idx, :])
            return [nms_scores, nms_class, anchors]

    def freeze_backbone(self):
        """Freeze backbone weights and bn layers."""
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
