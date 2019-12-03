import torch 
from models import EfficientDet

if __name__ == '__main__':
    inputs = torch.randn(4, 3, 512, 512)
    model = EfficientDet(levels=3)
    output = model(inputs)
    for p in output:
        print(p.size())