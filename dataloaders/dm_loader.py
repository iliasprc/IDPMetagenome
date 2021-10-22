import os

import torch
from torch.utils.data import Dataset

from .utils import read_data_
from random import randint
train_prefix = "train"
dev_prefix = "val"
test_prefix = "test"
train_filepath = "data/idp_seq_2_seq/train/all_train.txt"
dev_filepath = "data/idp_seq_2_seq/validation/all_valid.txt"
test_filepath = ""


class DMLoader(Dataset):
    def __init__(self, args, mode):
        cwd = args.cwd
        if mode == train_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.augment = True
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.mode = mode

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.augment:
                L = len(self.proteins[index])
                left = randint(0, L//4)
                right = randint(left+2, L)
                x = [self.w2i[amino] for amino in self.proteins[index]][left:right]
                y = [int(i) for i in self.annotations[index]][left:right]
                x = torch.LongTensor(x)  # .unsqueeze(-1)
                y = torch.LongTensor(y)  # .unsqueeze(-1)
                # print(x,y)
                return x, y

        x = torch.LongTensor([self.w2i[amino] for amino in self.proteins[index]])#.unsqueeze(-1)
        y = torch.LongTensor([int(i) for i in self.annotations[index]])#.unsqueeze(-1)
        #print(x,y)
        return x, y



class SSLDM(Dataset):
    def __init__(self, args, mode):
        cwd = args.cwd
        if mode == train_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.mode = mode

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):

        x = torch.LongTensor([self.w2i[amino] for amino in self.proteins[index]])#.unsqueeze(-1)
        #y = torch.FloatTensor([int(i) for i in self.annotations[index]])#.unsqueeze(-1)
        return x, x
