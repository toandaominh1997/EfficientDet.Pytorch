import torch
import torch.nn as nn
from torch.autograd import Variable
from models.efficientnet import EfficientNet
from models.bifpn import BiFPN
from layers.functions import PriorBox
from data import voc, coco
from .bifpn_v2 import BIFPN


class EfficientDet(nn.Module):
    def __init__(self,
                 num_class=21,
                 levels=3,
                 num_channels=128,
                 model_name='efficientnet-b0'):
        super(EfficientDet, self).__init__()
        self.num_class = num_class
        self.levels = levels
        self.num_channels = num_channels
        self.efficientnet = EfficientNet.from_pretrained(model_name)

        self.cfg = (coco, voc)[num_class == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.num_anchor = 9
        self.class_module = list()
        self.regress_module = list()
        for _ in range(3, 8):
            self.class_module.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.num_channels, out_channels=64,
                              kernel_size=2, stride=1),
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchor * num_class, kernel_size=2, stride=1)
                )
            )
            self.regress_module.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.num_channels, out_channels=64,
                              kernel_size=2, stride=1),
                    nn.Conv2d(
                        in_channels=64, out_channels=self.num_anchor * 4, kernel_size=2, stride=1)
                )
            )
            self.BIFPN = BIFPN(in_channels=[40, 80, 112, 192, 320],
                                out_channels=self.num_channels,
                                num_outs=5)
            self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        P1, P2, P3, P4, P5, P6, P7 = self.efficientnet(inputs)
        P3, P4, P5, P6, P7 = self.BIFPN([P3, P4, P5, P6, P7])
        feature_classes = []
        feature_bboxes = []
        for i, p in enumerate([P3, P4, P5, P6, P7]):
            feature_class = self.class_module[i](p)
            feature_class = feature_class.view(-1, self.num_class)
            feature_class = self.sigmoid(feature_class)
            feature_classes.append(feature_class)

            feature_bbox = self.regress_module[i](p)
            feature_bbox = feature_bbox.view(-1, 4)
            feature_bbox = self.sigmoid(feature_bbox)
            feature_bboxes.append(feature_bbox)
        feature_classes = torch.cat(feature_classes, axis=0)
        feature_bboxes = torch.cat(feature_bboxes, axis=0)

        output = (
            feature_bboxes.view(inputs.size(0), -1, 4),
            feature_classes.view(inputs.size(0), -1, self.num_class),
            self.priors
        )
        return output


