import torch 
import cv2 
from PIL import Image
from models import EfficientDet
from torchvision import transforms
import numpy as np 
import skimage
from datasets import get_augumentation, VOC_CLASSES

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
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model = self.model.cuda()

        if(self.weights is not None):
            print('Load pretrained Model')
            state = torch.load(self.weights, map_location=lambda storage, loc: storage)
            state_dict = state['state_dict']
            num_class = state['num_class']
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
            # print('scores: ', scores)
            idxs = np.where(scores.cpu().data.numpy()>0.25)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = VOC_CLASSES[int(classification[idxs[0][j]])]
                cv2.rectangle(show_image, (x1, y1), (x2, y2), (77, 255, 9), 3, 1)
                cv2.putText(show_image, label_name, (x1-10,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite('docs/output.png', show_image)

if __name__=='__main__':
    detect = Detect(weights = './weights/checkpoint_108.pth')
    output = detect.process('/root/data/VOCdevkit/VOC2007/JPEGImages/001999.jpg')
