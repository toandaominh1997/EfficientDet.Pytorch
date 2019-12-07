import torch
import torch.nn as nn
import math
from models.efficientnet import EfficientNet
from models.bifpn import BIFPN
from models.module import RegressionModel, ClassificationModel, Anchors, ClipBoxes, BBoxTransform
from torchvision.ops import nms 

MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
}
class EfficientDet(nn.Module):
    def __init__(self,
                 num_classes,
                 levels=3,
                 num_channels=128,
                 model_name='efficientdet-d0',
                 is_training=True,
                 threshold=0.5):
        super(EfficientDet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(MODEL_MAP[model_name])
        self.is_training = is_training
        self.BIFPN = BIFPN(in_channels=self.efficientnet.get_list_features()[2:],
                                out_channels=256,
                                num_outs=5)
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.threshold=threshold

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.freeze_bn()

    def forward(self, inputs):
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
            scores_over_thresh = (scores>self.threshold)[0, :, 0]

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
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()