import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import os
from skimage.io import imread


class UCF101_test(data.Dataset):
    def __init__(self, args):
        self.sequence_list = []
        temp_list = os.listdir(args.dataset_root) # ex) '/hdd/ucf101_interp_ours'
        for seq in temp_list:
            self.sequence_list.append('%s/%s'%(args.dataset_root, seq))

    def transform(self, frame1, frame2, frame3):
        return map(TF.to_tensor, (frame1, frame2, frame3))

    def __getitem__(self, index):
        first_fn = os.path.join(self.sequence_list[index],'frame_00.png')
        second_fn = os.path.join(self.sequence_list[index],'frame_01_gt.png')
        third_fn = os.path.join(self.sequence_list[index],'frame_02.png')

        frame1 = imread(first_fn)
        frame2 = imread(second_fn)
        frame3 = imread(third_fn)

        frame1, frame2, frame3 = self.transform(frame1, frame2, frame3)

        Input = torch.cat((frame1, frame3), dim=0)

        return Input, frame2, os.path.join(self.sequence_list[index],'frame_01.png')

    def __len__(self):
        return len(self.sequence_list)

