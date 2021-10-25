import os

import torch
from torch.utils.data import Dataset

from .utils import read_data_
from random import randint
train_prefix = "train"
dev_prefix = "val"
test_prefix = "test"

class DMLoader(Dataset):
    def __init__(self, args, mode):
        train_filepath = "data/idp_seq_2_seq/train/all_train.txt"
        dev_filepath = "data/idp_seq_2_seq/validation/ldr_valid.txt"
        test_filepath = ""

        cwd = args.cwd
        if mode == train_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.augment = True
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, dev_filepath))
            self.mode = mode
            self.augment = False
        self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        # if self.mode == 'train':
        #     if self.augment:
        #         L = len(self.proteins[index])
        #         left = randint(0, L//4)
        #         right = randint(L//2, L)
        #         x = [self.w2i[amino] for amino in self.proteins[index]][left:right]
        #         y = [int(i) for i in self.annotations[index]][left:right]
        #         x = torch.LongTensor(x)  # .unsqueeze(-1)
        #         y = torch.LongTensor(y)  # .unsqueeze(-1)
        #         x = torch.nn.functional.one_hot(x, num_classes=20).float()
        #
        #         #print(y.shape, y1.shape)
        #         #print(x[:5] , y1[:5])
        #         # print(x,y)
        #         return x, y#1

        x = torch.LongTensor([self.w2i[amino] for amino in self.proteins[index]])#.unsqueeze(-1)
        y = torch.LongTensor([int(i) for i in self.annotations[index]])#.unsqueeze(-1)
        assert x.shape == y.shape,print(self.names[index])
        y1 = torch.nn.functional.one_hot(y,num_classes=2)
        #x = torch.nn.functional.one_hot(x, num_classes=20).float()
        #print(y,y1)
        #print(x,y)
        return x, y#1



class SSLDM(Dataset):
    def __init__(self, args, mode):
        cwd = args.cwd
        train_filepath = "data/idp_seq_2_seq/train/all_train.txt"
        dev_filepath = "data/idp_seq_2_seq/validation/all_valid.txt"
        test_filepath = ""

        if mode == train_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, dev_filepath))
            self.mode = mode

        self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        indixes = list(range(len(self.classes)))
        print(self.classes)
        self.w2i = dict(zip(self.classes,indixes))
        print('classes\n\n', self.classes,len(self.classes))
    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        if self.mode == 'train':

            L = len(self.proteins[index])
            left = randint(0, L//4)
            right = randint(L//2, L)
            x = [self.w2i[amino] for amino in self.proteins[index]][left:right]
            y = [int(i) for i in self.annotations[index]][left:right]
            x = torch.LongTensor(x)  # .unsqueeze(-1)
            y = torch.LongTensor(y)  # .unsqueeze(-1)
           # x = torch.nn.functional.one_hot(x, num_classes=20).float()

            #print(y.shape, y1.shape)
            #print(x[:5] , y1[:5])
            # print(x,y)
            return x, x#1
        x = torch.LongTensor([self.w2i[amino] for amino in self.proteins[index]])#.unsqueeze(-1)
        #x = torch.nn.functional.one_hot(x, num_classes=20).float()
        #y = torch.FloatTensor([int(i) for i in self.annotations[index]])#.unsqueeze(-1)
        return x, x





class MXD494Loader(Dataset):
    def __init__(self, args, mode):
        train_prefix = "train"
        dev_prefix = "val"
        test_prefix = "test"
        train_filepath = "data/idp_seq_2_seq/mxd494/MXD494_train_all.txt"

        test_filepath = "data/idp_seq_2_seq/mxd494/MXD494.txt"
        dev_filepath = test_filepath
        cwd = args.cwd
        if mode == train_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.augment = True
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, dev_filepath))
            self.mode = mode
            self.augment = False
        self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        indixes = list(range(len(self.classes)))
        print(self.classes)
        self.w2i = dict(zip(self.classes,indixes))
        print('classes\n\n', self.classes,len(self.classes))

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        # if self.mode == 'train':
        #     if self.augment:
        #         L = len(self.proteins[index])
        #         left = randint(0, L//4)
        #         right = randint(L//2, L)
        #         x = [self.w2i[amino] for amino in self.proteins[index]][left:right]
        #         y = [int(i) for i in self.annotations[index]][left:right]
        #         x = torch.LongTensor(x)  # .unsqueeze(-1)
        #         y = torch.LongTensor(y)  # .unsqueeze(-1)
        #         x = torch.nn.functional.one_hot(x, num_classes=20).float()
        #
        #         #print(y.shape, y1.shape)
        #         #print(x[:5] , y1[:5])
        #         # print(x,y)
        #         return x, y#1
        x = [self.w2i[amino] for amino in self.proteins[index]]
        y = [int(i) for i in self.annotations[index]]
       # print(len(x),len(y),len(self.proteins[index]),len(self.annotations[index]))
        if len(x)!=len(y):
            print(self.names[index],'\n',self.proteins[index],'\n',self.annotations[index])
        x = torch.LongTensor([self.w2i[amino] for amino in self.proteins[index]])#.unsqueeze(-1)
        y = torch.LongTensor([int(i) for i in self.annotations[index]])#.unsqueeze(-1)
        #print(x.shape,y.shape)
        assert x.shape==y.shape,print(self.names[index])
        y1 = torch.nn.functional.one_hot(y,num_classes=2)
        #x = torch.nn.functional.one_hot(x, num_classes=20).float()
        #print(y,y1)
        #print(x,y)
        return x, y#1
