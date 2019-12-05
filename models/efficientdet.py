import torch
import torch.nn as nn

from models.efficientnet import EfficientNet
from models.bifpn_v2 import BIFPN
from models.module import RegressionModel, ClassificationModel, Anchors, ClipBoxes, BBoxTransform
from torchvision.ops import nms 

class EfficientDet(nn.Module):
    def __init__(self,
                 num_classes=21,
                 levels=3,
                 num_channels=128,
                 model_name='efficientnet-b0',
                 is_training=True):
        super(EfficientDet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(model_name)
        self.is_training = is_training
        self.BIFPN = BIFPN(in_channels=[40, 80, 112, 192, 320],
                                out_channels=256,
                                num_outs=5)
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
         
    def forward(self, inputs):
        print('inputs: ', inputs)
        features = self.efficientnet(inputs)
        features = self.BIFPN(features[2:])
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(inputs)
        if self.is_training:
            return classification, regression, anchors
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, inputs)
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores>0.05)[0, :, 0]
            if scores_over_thresh.sum() == 0:
                print('No boxes to NMS')
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, 0], iou_threshold = 0.5)
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
