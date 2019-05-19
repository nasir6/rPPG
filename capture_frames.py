import cv2
import numpy as np
import torch
from torch import nn
from models import LinkNet34
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageFilter
import time
import sys

class CaptureFrames():

    def __init__(self, bs, source, show_mask=False):
        self.frame_counter = 0
        self.batch_size = bs
        self.stop = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load('linknet.pth'))
        self.model.eval()
        self.model.to(self.device)
        self.show_mask = show_mask
        
    def __call__(self, pipe, source):
        self.pipe = pipe
        self.capture_frames(source)
  
    def capture_frames(self, source):
        
        img_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        camera = cv2.VideoCapture(source)
        time.sleep(1)
        self.model.eval()
        (grabbed, frame) = camera.read()

        time_1 = time.time()
        self.frames_count = 0
        while grabbed:
            (grabbed, orig) = camera.read()
            if not grabbed:
                continue
            
            shape = orig.shape[0:2]
            frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(256,256), cv2.INTER_LINEAR )
            
            
            k = cv2.waitKey(1)
            if k != -1:
                self.terminate(camera)
                break

            a = img_transform(Image.fromarray(frame))
            a = a.unsqueeze(0)
            imgs = Variable(a.to(dtype=torch.float, device=self.device))
            pred = self.model(imgs)
            
            pred= torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
            mask = pred.data.cpu().numpy()
            mask = mask.squeeze()
            
            # im = Image.fromarray(mask)
            # im2 = im.filter(ImageFilter.MinFilter(3))
            # im3 = im2.filter(ImageFilter.MaxFilter(5))
            # mask = np.array(im3)
            
            mask = mask > 0.8
            orig[mask==0]=0
            self.pipe.send([orig])

            if self.show_mask:
                cv2.imshow('mask', orig)
            
            if self.frames_count % 30 == 29:
                time_2 = time.time()
                sys.stdout.write(f'\rFPS: {30/(time_2-time_1)}')
                sys.stdout.flush()
                time_1 = time.time()


            self.frames_count+=1

        self.terminate(camera)

    
    def terminate(self, camera):
        self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()
        



