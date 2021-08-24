import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import os
from skimage.io import imread

class SNU_FILM_All_test(data.Dataset):
    def __init__(self,args):       
        self.first_list = list()
        self.second_list = list()
        self.target_list = list()
        self.third_list = list()
        
        data_list = os.listdir(args.dataset_root + '/test')

        if not os.path.isdir(os.path.join(args.dataset_root, args.name)):
            os.mkdir(os.path.join(args.dataset_root, args.name))
        for level in ['easy', 'medium', 'hard','extreme']:
            if not os.path.isdir('%s/%s' % (os.path.join(args.dataset_root, args.name), level)):
                os.mkdir('%s/%s' % (os.path.join(args.dataset_root, args.name), level))
            for data in data_list:
                if not os.path.isdir('%s/%s/%s' % (os.path.join(args.dataset_root, args.name), level,data)):
                    os.mkdir('%s/%s/%s' % (os.path.join(args.dataset_root, args.name), level,data))

                temp_seq_list = os.listdir(args.dataset_root + '/test/%s' % data)
                for temp in temp_seq_list:
                    if not os.path.isdir('%s/%s/%s/%s' % (os.path.join(args.dataset_root, args.name), level,data, temp)):
                        os.mkdir('%s/%s/%s/%s' % (os.path.join(args.dataset_root, args.name), level,data, temp))

            with open('%s/test-%s.txt' % (args.dataset_root, level), 'r') as txt:
                sequence_list = [line.strip() for line in txt]

            for seq in sequence_list:
                first, second, third = seq.split(' ')
                second = '%s/%s' % (args.dataset_root, second.replace('data/SNU-FILM/', ''))
                self.first_list.append('%s/%s' % (args.dataset_root, first.replace('data/SNU-FILM/', '')))
                self.target_list.append(second)
                self.second_list.append('%s' % (second.replace('/test/', '/%s/%s/' % (args.name,level))))
                self.third_list.append('%s/%s' % (args.dataset_root, third.replace('data/SNU-FILM/', '')))

    def transform(self, frame):
        return TF.to_tensor(frame)

    def __getitem__(self, index):
        frame1 = imread(self.first_list[index])
        frame2 = imread(self.first_list[index])
        frame3  = imread(self.third_list[index])

        frame1 = self.transform(frame1)
        frame2 = self.transform(frame2)
        frame3 = self.transform(frame3)
        
        Input = torch.cat((frame1,frame3), dim=0)

        return Input, frame2, self.second_list[index]

    def __len__(self):
        return len(self.first_list)