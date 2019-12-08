import torch 
import cv2 
from PIL import Image
from models import EfficientDet
from torchvision import transforms
import numpy as np 
import skimage
from datasets import get_augumentation, VOC_CLASSES
from timeit import default_timer as timer
import argparse


parser = argparse.ArgumentParser(description='EfficientDet')

parser.add_argument('-n', '--network', default='efficientdet-d0',
                    help='efficientdet-[d0, d1, ..]')
parser.add_argument('-s', '--score', default=True,
                    action="store_true", help='Show score')
parser.add_argument('-t', '--threshold', default=0.5,
                    type=float, help='Visualization threshold')
parser.add_argument('-w', '--weight', default='./weights/voc0712.pth',
                    type=str, help='Weight model path')
parser.add_argument('-c', '--cam',  default=True,
                    action="store_true", help='Use camera')
parser.add_argument('-f', '--file_name', default='pic.jpg',
                    help='Image path')

args = parser.parse_args()



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
        self.model = EfficientDet(
                    num_classes=num_class, model_name=args.network,
                    is_training=False, threshold=args.threshold
                    )
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model = self.model.cuda()

        if(self.weights is not None):
            print('Load pretrained Model')
            state = torch.load(self.weights, map_location=lambda storage, loc: storage)
            state_dict = state['state_dict']
            num_class = state['num_class']
            self.model.load_state_dict(state_dict)
        
        self.model.eval()

    def process(self, file_name=None, img=None, show=False):
        if file_name is not None:
            img = cv2.imread(file_name)

        show_aug = self.show_transform(image = img)
        show_image = show_aug['image']
        augmentation = self.transform(image = img)
        img = augmentation['image']
        img = img.to(self.device)
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            scores, classification, transformed_anchors = self.model(img)

            # idxs = np.where(scores.cpu().data.numpy()>args.threshold)

            for j in range(scores.shape[0]):
                bbox = transformed_anchors[[j], :][0]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = VOC_CLASSES[int(classification[[j]])]
                cv2.rectangle(show_image, (x1, y1), (x2, y2), (77, 255, 9), 3, 1)
                if args.score:
                    score = np.around(
                            scores[[j]].cpu().numpy(), decimals=2) * 100
                    cv2.putText(
                        show_image, '{} {}%'.format(label_name, int(score)),
                        (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2
                        )
                else:
                    cv2.putText(
                        show_image, label_name, (x1-10, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                        )
            if show:
                cv2.imshow("Detection", show_image)
                cv2.waitKey(0)
                cv2.imwrite('docs/output.png', show_image)
            else:
                return show_image

    def camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to open camera")
            exit(-1)
        count_tfps = 1
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:

            res, img = cap.read()
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1

            if accum_time > 1:
                accum_time = accum_time - 1
                fps = curr_fps
                curr_fps = 0
            if res:
                show_image = self.process(img=img)
                cv2.putText(
                    show_image, "FPS: " + str(fps), (10,  20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 250, 0), 2
                    )

                cv2.imshow("Detection", show_image)
                k = cv2.waitKey(1)
                if k == 27:
                    break
            else:
                print("Unable to read image")
                exit(-1)
            count_tfps += 1
        cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    detect = Detect(weights=args.weight)
    if args.cam:
        detect.camera()
    else:
        detect.process(file_name=args.file_name, show=True)
