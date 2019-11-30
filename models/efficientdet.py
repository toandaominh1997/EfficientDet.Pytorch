import torch 
import torch.nn as nn
from efficientnet import EfficientNet
from bifpn import BiFPN

class EfficientDet(nn.Module):
    def __init__(self,
                levels = 3,
                num_channels = 128,
                model_name = 'efficientnet-b0'):
        super(EfficientDet, self).__init__()
        self.levels = levels
        self.num_channels = num_channels
        self.efficientnet = EfficientNet.from_pretrained(model_name)
        self.bifpn = BiFPN(num_channels = self.num_channels)
    
    def forward(self, inputs):
        P1, P2, P3, P4, P5, P6, P7 = self.efficientnet(inputs)
        P3 = self.bifpn.Conv(in_channels=P3.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P3)
        P4 = self.bifpn.Conv(in_channels=P4.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P4)
        P5 = self.bifpn.Conv(in_channels=P5.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P5)
        P6 = self.bifpn.Conv(in_channels=P6.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P6)
        P7 = self.bifpn.Conv(in_channels=P7.size(1), out_channels=self.num_channels, kernel_size=1, stride=1, padding=0)(P7)
        for _ in range(self.levels):
            P3, P4, P5, P6, P7 = self.bifpn([P3, P4, P5, P6, P7])
        print(P3.size(), P4.size(), P5.size(), P6.size(), P7.size())


if __name__ =='__main__':
    inputs = torch.randn(4, 3, 640, 640)
    model = EfficientDet(levels=10)
    output = model(inputs)