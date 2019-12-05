import torch 
from models import EfficientDet

if __name__ == '__main__':
    inputs = torch.randn(2, 3, 512, 512)

    model = EfficientDet(num_classes=2, is_training=False)
    output = model(inputs)
    for p in output:
        print(p.size())
    