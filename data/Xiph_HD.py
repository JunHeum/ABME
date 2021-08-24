import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import os
from skimage.io import imread

class Xiph_HD_test(data.Dataset):
    def __init__(self,args):
        
        self.first_list = list()
        self.second_list = list()
        self.third_list = list()

        self.target_list = list()

        if not os.path.isdir(os.path.join(args.dataset_root, args.name)):
            os.mkdir(os.path.join(args.dataset_root, args.name))

        vid_list = os.listdir('%s/sequence_center'%args.dataset_root)
        for vid in vid_list:
            if not os.path.isdir(os.path.join(args.dataset_root, args.name,vid)):
                os.mkdir('%s/%s/%s'%(args.dataset_root, args.name, vid))

        with open('%s/seq_selected_d3.txt'%args.dataset_root,'r') as txt:
            seq_list = [line.strip() for line in txt]
        
        for seq in seq_list:
            first_fn, second_fn, third_fn = seq.split(' ')
            self.first_list.append('%s/sequence_center/%s'%(args.dataset_root,  first_fn))
            self.second_list.append('%s/%s/%s'%(args.dataset_root,args.name, second_fn))
            self.third_list.append('%s/sequence_center/%s'%(args.dataset_root,  third_fn))
            self.target_list.append('%s/sequence_center/%s'%(args.dataset_root,  second_fn))
    
    def transform(self, frame):
        return TF.to_tensor(frame)

    def __getitem__(self, index):
        frame1  = imread(self.first_list[index])
        frame2  = imread(self.target_list[index])
        frame3  = imread(self.third_list[index])

        frame1 = self.transform(frame1)
        frame2 = self.transform(frame2)
        frame3 = self.transform(frame3)
        
        Input = torch.cat((frame1,frame3), dim=0)

        return Input, frame2, self.second_list[index]

    def __len__(self):
        return len(self.first_list)