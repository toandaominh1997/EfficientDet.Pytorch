import torch 
from models import EfficientDet

if __name__ == '__main__':
    inputs = torch.randn(5, 3, 512, 512).cuda()

    model = EfficientDet(num_classes=2, is_training=True)
    model = model.cuda()
    output = model(inputs)
    for p in output:
        print(p.size())