import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import cv2
import random
import os
from skimage.io import imread

class Vimeo_train(data.Dataset):
    def __init__(self, args):
        self.crop_size = [256,256]
        self.sequence_list = []
        with open('%s/tri_trainlist.txt' % args.dataset_root, 'r') as txt:
            for line in txt:
                self.sequence_list.append('%s/sequences/%s' % (args.dataset_root, line.strip()))
        
    def transform(self, frame1, frame2, frame3):
        # Random cropping augmentation
        h_offset = random.choice(range(256 - self.crop_size[0] + 1))
        w_offset = random.choice(range(448 - self.crop_size[1]+ 1))

        frame1 = frame1[h_offset:h_offset + self.crop_size[0], w_offset: w_offset + self.crop_size[1], :]
        frame2 = frame2[h_offset:h_offset + self.crop_size[0], w_offset: w_offset + self.crop_size[1], :]
        frame3 = frame3[h_offset:h_offset + self.crop_size[0], w_offset: w_offset + self.crop_size[1], :]

        # Rotation augmentation
        if self.crop_size[0] == self.crop_size[1]:
            if random.randint(0, 1):
                frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
                frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
                frame3 = cv2.rotate(frame3, cv2.ROTATE_90_CLOCKWISE)
            elif random.randint(0, 1):
                frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
                frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
                frame3 = cv2.rotate(frame3, cv2.ROTATE_180)
            elif random.randint(0, 1):
                frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame3 = cv2.rotate(frame3, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Flip augmentation
        if random.randint(0, 1):
            flip_code = random.randint(-1,1) # 0 : Top-bottom | 1: Right-left | -1: both
            frame1 = cv2.flip(frame1, flip_code)
            frame2 = cv2.flip(frame2, flip_code)
            frame3 = cv2.flip(frame3, flip_code)

        # return map(TF.to_tensor, (frame1, frame2, frame3, flow, frame_fw, frame_bw))
        return map(TF.to_tensor, (frame1, frame2, frame3))

    def __getitem__(self, index):
        if random.randint(0,1):
            First_fn  = os.path.join(self.sequence_list[index], 'im1.png')
            Third_fn  = os.path.join(self.sequence_list[index], 'im3.png')
        else:
            First_fn  = os.path.join(self.sequence_list[index], 'im3.png')
            Third_fn  = os.path.join(self.sequence_list[index], 'im1.png')
        
        Second_fn = os.path.join(self.sequence_list[index], 'im2.png')

        frame1 = imread(First_fn)
        frame2 = imread(Second_fn)
        frame3 = imread(Third_fn)

        frame1, frame2, frame3 = self.transform(frame1, frame2, frame3)

        Input = torch.cat((frame1, frame3), dim=0)

        return Input, frame2

    def __len__(self):
        return len(self.sequence_list)

class Vimeo_validation(data.Dataset):
    def __init__(self, args):
        self.sequence_list = []
        with open('%s/tri_testlist.txt'%args.dataset_root,'r') as txt:
            for line in txt:
                self.sequence_list.append('%s/sequences/%s'%(args.dataset_root, line.strip()))

    def transform(self, frame1, frame2, frame3):
        return map(TF.to_tensor, (frame1, frame2, frame3))

    def __getitem__(self, index):
        first_fn = os.path.join(self.sequence_list[index],'im1.png')
        second_fn = os.path.join(self.sequence_list[index],'im2.png')
        third_fn = os.path.join(self.sequence_list[index],'im3.png')

        frame1 = imread(first_fn)
        frame2 = imread(second_fn)
        frame3 = imread(third_fn)

        frame1, frame2, frame3 = self.transform(frame1, frame2, frame3)

        Input = torch.cat((frame1, frame3), dim=0)

        return Input, frame2

    def __len__(self):
        return len(self.sequence_list)

class Vimeo_test(data.Dataset):
    def __init__(self, args):
        self.sequence_list = []
        with open('%s/tri_testlist.txt'%args.dataset_root,'r') as txt:
            for line in txt:
                self.sequence_list.append('%s/input/%s'%(args.dataset_root, line.strip()))

        if not os.path.isdir('%s/%s'%(args.dataset_root, args.name)):
            os.mkdir('%s/%s'%(args.dataset_root, args.name))
        if len(os.listdir('%s/%s'%(args.dataset_root, args.name))) != 78:
            for seq in self.sequence_list:
                idx1, idx2 = seq.split('/')[-2], seq.split('/')[-1]
                if not os.path.isdir('%s/%s/%s'%(args.dataset_root, args.name,idx1)):
                    os.mkdir('%s/%s/%s'%(args.dataset_root, args.name, idx1))
                if not os.path.isdir('%s/%s/%s/%s'%(args.dataset_root, args.name,idx1,idx2)):
                    os.mkdir('%s/%s/%s/%s'%(args.dataset_root, args.name, idx1, idx2))

    def transform(self, frame1, frame2, frame3):
        return map(TF.to_tensor, (frame1, frame2, frame3))

    def __getitem__(self, index):
        first_fn  = os.path.join(self.sequence_list[index],'im1.png')
        second_fn = os.path.join(self.sequence_list[index].replace('input','target'),'im2.png')
        third_fn  = os.path.join(self.sequence_list[index],'im3.png')

        frame1 = imread(first_fn)
        frame2 = imread(second_fn)
        frame3 = imread(third_fn)

        frame1, frame2, frame3 = self.transform(frame1, frame2, frame3)

        Input = torch.cat((frame1, frame3), dim=0)

        return Input, frame2, second_fn

    def __len__(self):
        return len(self.sequence_list)
