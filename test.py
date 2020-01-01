import torch 
from models import EfficientDet

if __name__ == '__main__':
    inputs = torch.randn(5, 3, 512, 512)

    model = EfficientDet(num_classes=2, is_training=False)
    model = model
    output = model(inputs)
    for out in output:
        print(out.size())
#     print('output: ', output)