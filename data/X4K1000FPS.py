import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import os
from skimage.io import imread

class X4K1000FPS_test(data.Dataset):
    def __init__(self,args):

        if args.is_save:
            if not os.path.isdir(os.path.join(args.dataset_root, args.name)):
                os.mkdir(os.path.join(args.dataset_root, args.name))
                os.mkdir('%s/%s/Type1'%(args.dataset_root, args.name))
                os.mkdir('%s/%s/Type2'%(args.dataset_root, args.name))
                os.mkdir('%s/%s/Type3'%(args.dataset_root, args.name))
                
        self.first_list = list()
        self.third_list = list()
        self.seq_fn_list = list()
        
        with open('%s/sequence_list.txt'%args.dataset_root,'r') as txt:
            seq_list = [line.strip() for line in txt]
        
        for seq in seq_list:
            first, third, seq_fn, _ = seq.split()

            self.first_list.append(os.path.join(args.dataset_root,first)) # idx 0
            self.third_list.append(os.path.join(args.dataset_root,third)) # idx 32
            
            seq_fn = os.path.join(args.dataset_root,args.name,seq_fn)
            self.seq_fn_list.append(seq_fn) # Type1/TEST01_003_f0433

            if not os.path.isdir(seq_fn):
                os.mkdir(seq_fn)
                   
    def transform(self, frame):
        return TF.to_tensor(frame)

    def __getitem__(self, index):
        frame1  = imread(self.first_list[index])
        frame3  = imread(self.third_list[index])

        frame1 = self.transform(frame1)
        frame3 = self.transform(frame3)
        
        Input = torch.cat((frame1,frame3), dim=0)

        return Input, self.seq_fn_list[index]

    def __len__(self):
        return len(self.first_list)