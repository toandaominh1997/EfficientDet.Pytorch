import torch 
import cv2 
from PIL import Image
from models import EfficientDet
from torchvision import transforms
import numpy as np 
import skimage
class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, side=512):
        rows, cols, cns = image.shape

        scale = float(side)/float(max(rows, cols))
        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = side-rows
        pad_h = side-cols
        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        return torch.from_numpy(new_image)
class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, image):
        return (image.astype(np.float32)-self.mean)/self.std

class Detect(object):
    """
        dir_name: Folder or image_file
    """
    def __init__(self, weights, num_class=21):
        super(Detect,  self).__init__()
        self.weights = weights
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            Normalizer(),
            Resizer()
        ])
        self.model = EfficientDet(num_classes=num_class, is_training=False)
        self.model = self.model.to(self.device)
        if(self.weights is not None):
            print('Load pretrained Model')
            state_dict = torch.load(weights)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()

    def process(self, file_name):
        img = cv2.imread(file_name)
        cv2.imwrite('kaka.png', img)
        img = self.transform(img)
        img = img.to(self.device)
        img = img.unsqueeze(0).permute(0, 3, 1, 2)
        scores, classification, transformed_anchors = self.model(img)
        print('scores: ', scores)
        scores = scores.detach().cpu().numpy()
        idxs = np.where(scores>0.1)
        return idxs

if __name__=='__main__':
    detect = Detect(weights = './weights/checkpoint_87.pth')
    output = detect.process('/root/data/VOCdevkit/VOC2007/JPEGImages/001234.jpg')
    print('output: ', output)
