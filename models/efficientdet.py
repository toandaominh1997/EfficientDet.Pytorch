import torch 
import torch.nn as nn
from efficientnet import EfficientNet
from bifpn import BiFPN

class EfficientDet(nn.Module):
    def __init__(self,
                num_class = 10,
                levels = 3,
                num_channels = 128,
                model_name = 'efficientnet-b0'):
        super(EfficientDet, self).__init__()
        self.num_class = num_class 
        self.levels = levels
        self.num_channels = num_channels
        self.efficientnet = EfficientNet.from_pretrained(model_name)
        self.bifpn = BiFPN(num_channels = self.num_channels)

        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        
    
    def forward(self, inputs):
        P1, P2, P3, P4, P5, P6, P7 = self.efficientnet(inputs)
        P3 = self.bifpn.Conv(in_channels=P3.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P3)
        P4 = self.bifpn.Conv(in_channels=P4.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P4)
        P5 = self.bifpn.Conv(in_channels=P5.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P5)
        P6 = self.bifpn.Conv(in_channels=P6.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P6)
        P7 = self.bifpn.Conv(in_channels=P7.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P7)
        for _ in range(self.levels):
            P3, P4, P5, P6, P7 = self.bifpn([P3, P4, P5, P6, P7])
        P = [P3, P4, P5, P6, P7]
        features_class = [self.class_net(p, self.num_class) for p in P]
        features_class = torch.cat(features_class, axis=0)
        features_bbox = [self.regression_net(p) for p in P]
        features_bbox = torch.cat(features_bbox, axis=0)
        output = (
                features_bbox.view(loc.size(0), -1, 4),
                features_class.view(conf.size(0), -1, self.num_class),
                self.priors
            )
        print('class: {}, bbox: {}'.format(features_class.size(), features_bbox.size()))
        
    @staticmethod
    def class_net(features, num_class, num_anchor=9):
        features = nn.Sequential(
            nn.Conv2d(in_channels=features.size(1), out_channels=features.size(2), kernel_size = 3, stride=1),
            nn.Conv2d(in_channels=features.size(2), out_channels=num_anchor*num_class, kernel_size = 3, stride=1)
        )(features)
        features = features.view(-1, num_class)
        features = nn.Sigmoid()(features)
        return features 
    @staticmethod
    def regression_net(features, num_anchor=9):
        features = nn.Sequential(
            nn.Conv2d(in_channels=features.size(1), out_channels=features.size(2), kernel_size = 3, stride=1),
            nn.Conv2d(in_channels=features.size(2), out_channels=num_anchor*4, kernel_size = 3, stride=1)
        )(features)
        features = features.view(-1, 4)
        features = nn.Sigmoid()(features)
        return features 


if __name__ =='__main__':
    inputs = torch.randn(4, 3, 640, 640)
    model = EfficientDet(levels=10)
    output = model(inputs)