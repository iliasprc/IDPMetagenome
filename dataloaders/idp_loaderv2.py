import logging
import os
from typing import Union, List, Tuple, Sequence, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import TAPETokenizer
from .utils.utils import read_data_

train_prefix = "train"
dev_prefix = "val"
test_prefix = "test"

logger = logging.getLogger(__name__)


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class DM_idp_dataset(Dataset):
    def __init__(self, cwd : str, split: str, tokenizer: Union[str, TAPETokenizer] = 'unirep', ):
        train_filepath = "data/idp_seq_2_seq/train/all_train.txt"
        dev_filepath = "data/idp_seq_2_seq/validation/ldr_valid.txt"

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if split == train_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.augment = True
            self.split = split
        elif split == dev_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, dev_filepath))
            self.split = split
            self.augment = False
        self.classes = self.tokenizer.tokens
        self._num_examples = len(self.proteins)

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):

        item = self.proteins[index]
        token_ids = self.tokenizer.encode(item)
        input_mask = np.ones_like(token_ids)
        labels = np.asarray([int(i) for i in self.annotations[index]])
#        assert token_ids.shape == labels.shape, print(self.names[index])
        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, -1))

        output = {'input_ids' : input_ids,
                  'input_mask': input_mask,
                  'targets'   : ss_label}

        return output

#
# class SSLDM(Dataset):
#     def __init__(self, args, mode):
#         cwd = args.cwd
#         train_filepath = "data/idp_seq_2_seq/train/all_train.txt"
#         dev_filepath = "data/idp_seq_2_seq/validation/all_valid.txt"
#         test_filepath = ""
#
#         if mode == train_prefix:
#             self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
#                 os.path.join(cwd, train_filepath))
#             self.mode = mode
#         elif mode == dev_prefix:
#             self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
#                 os.path.join(cwd, dev_filepath))
#             self.mode = mode
#
#         self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
#         'V', 'W',
#                     'X', 'Y']
#     def __len__(self):
#         return len(self.proteins)
#
#     def __getitem__(self, index):
#         if self.mode == 'train':
#
#             L = len(self.proteins[index])
#             left = randint(0, L//4)
#             right = randint(L//2, L)
#             x = [self.w2i[amino] for amino in self.proteins[index]][left:right]
#             y = [int(i) for i in self.annotations[index]][left:right]
#             x = torch.LongTensor(x)  # .unsqueeze(-1)
#             y = torch.LongTensor(y)  # .unsqueeze(-1)
#            # x = torch.nn.functional.one_hot(x, num_classes=20).float()
#
#             #print(y.shape, y1.shape)
#             #print(x[:5] , y1[:5])
#             # print(x,y)
#             return x, x#1
#         x = torch.LongTensor([self.w2i[amino] for amino in self.proteins[index]])#.unsqueeze(-1)
#         #x = torch.nn.functional.one_hot(x, num_classes=20).float()
#         #y = torch.FloatTensor([int(i) for i in self.annotations[index]])#.unsqueeze(-1)
#         return x, x
#
#
#
#
#
# class MXD494Loader(Dataset):
#     def __init__(self, args, mode):
#         train_prefix = "train"
#         dev_prefix = "val"
#         test_prefix = "test"
#         train_filepath = "data/idp_seq_2_seq/mxd494/MXD494_train_all.txt"
#
#         test_filepath = "data/idp_seq_2_seq/mxd494/MXD494.txt"
#         dev_filepath = test_filepath
#         cwd = args.cwd
#         if mode == train_prefix:
#             self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
#                 os.path.join(cwd, train_filepath))
#             self.augment = True
#             self.mode = mode
#         elif mode == dev_prefix:
#             self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
#                 os.path.join(cwd, dev_filepath))
#             self.mode = mode
#             self.augment = False
#         self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
#         'V', 'W', 'X', 'Y']
#     def __len__(self):
#         return len(self.proteins)
#
#     def __getitem__(self, index):
#         # if self.mode == 'train':
#         #     if self.augment:
#         #         L = len(self.proteins[index])
#         #         left = randint(0, L//4)
#         #         right = randint(L//2, L)
#         #         x = [self.w2i[amino] for amino in self.proteins[index]][left:right]
#         #         y = [int(i) for i in self.annotations[index]][left:right]
#         #         x = torch.LongTensor(x)  # .unsqueeze(-1)
#         #         y = torch.LongTensor(y)  # .unsqueeze(-1)
#         #         x = torch.nn.functional.one_hot(x, num_classes=20).float()
#         #
#         #         #print(y.shape, y1.shape)
#         #         #print(x[:5] , y1[:5])
#         #         # print(x,y)
#         #         return x, y#1
#
#         x = torch.LongTensor([self.w2i[amino] for amino in self.proteins[index]])#.unsqueeze(-1)
#         y = torch.LongTensor([int(i) for i in self.annotations[index]])#.unsqueeze(-1)
#         y1 = torch.nn.functional.one_hot(y,num_classes=2)
#         #x = torch.nn.functional.one_hot(x, num_classes=20).float()
#         #print(y,y1)
#         #print(x,y)
#         return x, y#1
