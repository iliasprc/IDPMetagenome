import os

import torch
from torch.utils.data import Dataset

from .utils import read_data_

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
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.mode = mode

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):

        x = torch.FloatTensor([self.w2i[amino] for amino in self.proteins[index]])
        y = torch.FloatTensor([int(i) for i in self.annotations[index]])
        return x, y
