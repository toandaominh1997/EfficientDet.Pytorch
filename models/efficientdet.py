import torch
import torch.nn as nn
import math
from models.efficientnet import EfficientNet
from models.bifpn import BIFPN
from .retinahead import RetinaHead
from .compute_loss import generate_anchors, decode
from models.module import RegressionModel, ClassificationModel, Anchors, ClipBoxes, BBoxTransform
from torchvision.ops import nms 

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
                 network = 'efficientdet-d0',
                 D_bifpn=3,
                 W_bifpn=88,
                 D_class=3,
                 is_training=True,
                 threshold=0.5,
                 iou_threshold=0.5,
                 top_n = 1000):
        super(EfficientDet, self).__init__()
        self.is_training = is_training
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.top_n = top_n
        self.ratios = [1.0, 2.0, 0.5]
        self.scales = [4 * 2**(i/3) for i in range(3)]
        self.anchors = {}
        self.backbone = EfficientNet.from_pretrained(MODEL_MAP[network])
        self.neck = BIFPN(in_channels=self.backbone.get_list_features()[-5:],
                                out_channels=W_bifpn,
                                stack=D_bifpn,
                                num_outs=5)
        self.bbox_head = RetinaHead(num_classes = num_classes,
                                    in_channels = W_bifpn)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()
        
    def forward(self, inputs):
        x = self.extract_feat(inputs)
        outs = self.bbox_head(x)
        cls_heads= [out for out in outs[0]]
        box_heads = [out for out in outs[1]]
        if self.is_training:
            return cls_heads, box_heads
        cls_heads = [cls_head.sigmoid() for cls_head in cls_heads]
        
        decoded = []
        for cls_head, box_head in zip(cls_heads, box_heads):
            stride = inputs.shape[-1]//cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)
                decoded.append(decode(cls_head, box_head, stride, self.threshold, self.top_n, self.anchors[stride]))
        
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        anchor_nms_idx = nms(decoded[1][0, :, :], decoded[0][0, :], iou_threshold = self.iou_threshold)
        
        return decoded[0][0, anchor_nms_idx], decoded[2][0, anchor_nms_idx], decoded[1][0, anchor_nms_idx, :]
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x[-5:])
        return x