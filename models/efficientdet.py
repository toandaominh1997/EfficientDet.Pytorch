import torch
import torch.nn as nn
import math
from models.efficientnet import EfficientNet
from models.bifpn import BIFPN
from .retinahead import RetinaHead

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
                 network = 'efficientdet-d0',
                 D_bifpn=3,
                 W_bifpn=88,
                 D_class=3,
                 is_training=True,
                 threshold=0.5,
                 iou_threshold=0.5):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNet.from_pretrained(MODEL_MAP[network])
        self.is_training = is_training
        self.neck = BIFPN(in_channels=self.backbone.get_list_features()[-5:],
                                out_channels=W_bifpn,
                                stack=D_bifpn,
                                num_outs=5)
        self.bbox_head = RetinaHead(num_classes = num_classes,
                                    in_channels = W_bifpn)


    def forward(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs 
    
    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x)
        return x