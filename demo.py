import torch 
import cv2 
from PIL import Image
from models import EfficientDet
from torchvision import transforms
import numpy as np 
import skimage
from datasets import get_augumentation

class Detect(object):
    """
        dir_name: Folder or image_file
    """
    def __init__(self, weights, num_class=21):
        super(Detect,  self).__init__()
        self.weights = weights
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.transform = get_augumentation(phase='test')
        self.show_transform = get_augumentation(phase='show')
        self.model = EfficientDet(num_classes=num_class, is_training=False)
        self.model = self.model.to(self.device)
        if(self.weights is not None):
            print('Load pretrained Model')
            state_dict = torch.load(weights)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()

    def process(self, file_name):
        img = cv2.imread(file_name)

        show_aug = self.show_transform(image = img)
        show_image = show_aug['image']
        augmentation = self.transform(image = img)
        img = augmentation['image']
        img = img.to(self.device)
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            scores, classification, transformed_anchors = self.model(img)
            for i in range(transformed_anchors.size(0)):
                bbox = transformed_anchors[i, :]  
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                print(x1, x2, y1, y2)
                color = (255, 0, 0)
                thickness = 2
                cv2.rectangle(show_image, (x1, y1), (x2, y2), color, thickness)
            cv2.imwrite('output.png', show_image)

if __name__=='__main__':
    detect = Detect(weights = './weights/checkpoint_100.pth')
    output = detect.process('/root/data/VOCdevkit/VOC2007/JPEGImages/003476.jpg')
    print('output: ', output)
