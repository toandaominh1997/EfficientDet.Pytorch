import torch
import torch.nn as nn
from torch.autograd import Variable
from models.efficientnet import EfficientNet
from models.bifpn import BiFPN
from layers.functions import PriorBox
from data import voc, coco



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
        self.bifpn = BiFPN(num_channels=self.num_channels)
        self.cfg = (coco, voc)[num_class == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.num_anchor = 9

        self.input_image = 512
        self.Conv = [
            nn.Conv2d(in_channels=40,
                      out_channels=self.num_channels, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=80,
                      out_channels=self.num_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=112,
                      out_channels=self.num_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=192,
                      out_channels=self.num_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=320,
                      out_channels=self.num_channels, kernel_size=3, padding=1),
        ]
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

    def forward(self, inputs):

        P1, P2, P3, P4, P5, P6, P7 = self.efficientnet(inputs)
        P3 = self.Conv[0](P3)
        P4 = self.Conv[1](P4)
        P5 = self.Conv[2](P5)
        P6 = self.Conv[3](P6)
        P7 = self.Conv[4](P7)
        for _ in range(self.levels):
            P3, P4, P5, P6, P7 = self.bifpn([P3, P4, P5, P6, P7])
        feature_classes = []
        feature_bboxes = []
        for i, p in enumerate([P3, P4, P5, P6, P7]):
            feature_class = self.class_module[i](p)
            feature_class = feature_class.view(-1, self.num_class)
            feature_class = nn.Sigmoid()(feature_class)
            feature_classes.append(feature_class)

            feature_bbox = self.regress_module[i](p)
            feature_bbox = feature_bbox.view(-1, 4)
            feature_bbox = nn.Sigmoid()(feature_bbox)
            feature_bboxes.append(feature_bbox)
        feature_classes = torch.cat(feature_classes, axis=0)
        feature_bboxes = torch.cat(feature_bboxes, axis=0)

        output = (
            feature_bboxes.view(inputs.size(0), -1, 4),
            feature_classes.view(inputs.size(0), -1, self.num_class),
            self.priors
        )
        return output

if __name__ == '__main__':
    inputs = torch.randn(4, 3, 512, 512)
    model = EfficientDet(levels=3)
    output = model(inputs)
