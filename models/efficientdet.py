import torch
import torch.nn as nn

from models.efficientnet import EfficientNet
from models.bifpn_v2 import BIFPN
from models.module import RegressionModel, ClassificationModel, Anchors

class EfficientDet(nn.Module):
    def __init__(self,
                 num_classes=21,
                 levels=3,
                 num_channels=128,
                 model_name='efficientnet-b0'):
        super(EfficientDet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(model_name)

        self.BIFPN = BIFPN(in_channels=[40, 80, 112, 192, 320],
                                out_channels=256,
                                num_outs=5)
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.anchors = Anchors()
    def forward(self, inputs):

        features = self.efficientnet(inputs)
        features = self.BIFPN([features[3:]])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(inputs)
        return classification, regression, anchors 


