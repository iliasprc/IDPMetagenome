import os
import random
from random import randint

import torch
from torch.utils.data import Dataset

from .utils import read_data_

train_prefix = "train"
dev_prefix = "val"
test_prefix = "test"

class DMshort(Dataset):
    def __init__(self, config, mode):
        train_filepath = "data/idp_seq_2_seq/train/all_train.txt"
        dev_filepath = "data/idp_seq_2_seq/validation/all_valid.txt"
        test_filepath = ""

        cwd = config.cwd
        if mode == train_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.augment = True
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, _, _ = read_data_(
                os.path.join(cwd, dev_filepath))
            self.mode = mode
            self.augment = False
        self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                        'V', 'W', 'X', 'Y']

        indixes = list(range(len(self.classes)))
        print(self.classes)
        self.w2i = dict(zip(self.classes, indixes))
        # print('classes\n\n', self.classes,len(self.classes))
        self.ssl = config.dataset.type == 'SSL'
        self.use_elmo = config.dataset.use_elmo
        if self.use_elmo:
            print('\n USE ELMO \n')
            # model_dir = Path('/config/uniref50_v2')
            # weights = model_dir / 'weights.hdf5'
            # options = model_dir / 'options.json'
            # self.embedder = ElmoEmbedder(options, weights, cuda_device=-1)
            if self.ssl:
                print('\n SELF-SUPERVISED\n')
            else:
                print('\nIDP fully-supervised\n')

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

        if self.mode == 'train' and self.augment:

            L = len(self.proteins[index])
            left = randint(0, L // 4)
            right = randint(L // 2, L)
            x = [self.w2i[amino] for amino in self.proteins[index]]  # [left:right]
            if self.use_elmo:
                x = self.proteins[index][:1022]
                # print(seq)
                # x =  torch.FloatTensor(self.embedder.embed_sentence(list(seq))).sum(dim=0).cpu()
            else:
                x = torch.LongTensor(x)
            y = [int(i) for i in self.annotations[index][:1022]]  # [left:right]
            # x = torch.LongTensor(x)  # .unsqueeze(-1)
            y = torch.LongTensor(y)  # .unsqueeze(-1)
            # x = torch.nn.functional.one_hot(x, num_classes=20).float()

            # print(y.shape, y1.shape)
            # print(x[:5] , y1[:5])
            # print(x,y)
            if self.ssl:
                return x, x
            return x, y  # 1

        x = [self.w2i[amino] for amino in self.proteins[index]]
        y = [int(i) for i in self.annotations[index]]
        # print(len(x),len(y),len(self.proteins[index]),len(self.annotations[index]))
        if self.use_elmo:
            seq = self.proteins[index][:1022]
            # print(seq)
            x = seq
        else:
            x = torch.LongTensor(x)
        y = torch.LongTensor([int(i) for i in self.annotations[index]][:1022])  # .unsqueeze(-1)
        # print(x.shape,y.shape)
#        assert x.shape == y.shape, print(self.names[index])
        y1 = torch.nn.functional.one_hot(y, num_classes=2)
        # x = torch.nn.functional.one_hot(x, num_classes=20).float()
        # print(y,y1)
        # print(x,y)
        if self.ssl:
            return x, x
        return x, y  # 1


class DMLoader(Dataset):
    def __init__(self, config, mode):
        train_filepath = "data/idp_seq_2_seq/train/all_train.txt"
        dev_filepath = "data/idp_seq_2_seq/validation/all_valid.txt"
        test_filepath = ""

        cwd = config.cwd
        if mode == train_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.augment = True
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, _, _ = read_data_(
                os.path.join(cwd, dev_filepath))
            self.mode = mode
            self.augment = False
        self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                        'V', 'W', 'X', 'Y']

        indixes = list(range(len(self.classes)))
        print(self.classes)
        self.w2i = dict(zip(self.classes, indixes))
        # print('classes\n\n', self.classes,len(self.classes))
        self.ssl = config.dataset.type == 'SSL'
        self.use_elmo = config.dataset.use_elmo
        if self.use_elmo:
            print('\n USE ELMO \n')
            # model_dir = Path('/config/uniref50_v2')
            # weights = model_dir / 'weights.hdf5'
            # options = model_dir / 'options.json'
            # self.embedder = ElmoEmbedder(options, weights, cuda_device=-1)
            if self.ssl:
                print('\n SELF-SUPERVISED\n')
            else:
                print('\nIDP fully-supervised\n')

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

        if self.mode == 'train' and self.augment:

            L = len(self.proteins[index])
            left = randint(0, L // 4)
            right = randint(L // 2, L)
            x = [self.w2i[amino] for amino in self.proteins[index]]  # [left:right]
            if self.use_elmo:
                x = self.proteins[index]
                # print(seq)
                # x =  torch.FloatTensor(self.embedder.embed_sentence(list(seq))).sum(dim=0).cpu()
            else:
                x = torch.LongTensor(x)
            y = [int(i) for i in self.annotations[index]]  # [left:right]
            # x = torch.LongTensor(x)  # .unsqueeze(-1)
            y = torch.LongTensor(y)  # .unsqueeze(-1)
            # x = torch.nn.functional.one_hot(x, num_classes=20).float()

            # print(y.shape, y1.shape)
            # print(x[:5] , y1[:5])
            # print(x,y)
            if self.ssl:
                return x, x
            return x, y  # 1

        x = [self.w2i[amino] for amino in self.proteins[index]]
        y = [int(i) for i in self.annotations[index]]
        # print(len(x),len(y),len(self.proteins[index]),len(self.annotations[index]))
        if self.use_elmo:
            seq = self.proteins[index]
            # print(seq)
            x = seq
        else:
            x = torch.LongTensor(x)
        y = torch.LongTensor([int(i) for i in self.annotations[index]])  # .unsqueeze(-1)
        # print(x.shape,y.shape)
#        assert x.shape == y.shape, print(self.names[index])
        y1 = torch.nn.functional.one_hot(y, num_classes=2)
        # x = torch.nn.functional.one_hot(x, num_classes=20).float()
        # print(y,y1)
        # print(x,y)
        if self.ssl:
            return x, x
        return x, y  # 1


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
#         'V', 'W', 'X', 'Y']
#         indixes = list(range(len(self.classes)))
#         print(self.classes)
#         self.w2i = dict(zip(self.classes,indixes))
#         print('classes\n\n', self.classes,len(self.classes))
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
#             #y = [int(i) for i in self.annotations[index]][left:right]
#             x = torch.LongTensor(x)  # .unsqueeze(-1)
#             #y = torch.LongTensor(y)  # .unsqueeze(-1)
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


class MXD494Loader(Dataset):
    def __init__(self, config, mode):
        train_prefix = "train"
        dev_prefix = "val"
        test_prefix = "test"
        train_filepath = "data/idp_seq_2_seq/mxd494/MXD494_train_all.txt"

        test_filepath = "data/idp_seq_2_seq/mxd494/MXD494.txt"
        dev_filepath = test_filepath
        cwd = config.cwd

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
        self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                        'W', 'X', 'Y']

        indixes = list(range(len(self.classes)))
        print(self.classes)
        self.w2i = dict(zip(self.classes, indixes))
        # print('classes\n\n', self.classes,len(self.classes))
        self.ssl = config.dataset.type == 'SSL'
        self.use_elmo = config.dataset.use_elmo
        if self.use_elmo:
            print('\n USE ELMO \n')
            # model_dir = Path('/config/uniref50_v2')
            # weights = model_dir / 'weights.hdf5'
            # options = model_dir / 'options.json'
            # embedder = ElmoEmbedder(options, weights, cuda_device=0)
        if self.ssl:
            print('\n SELF-SUPERVISED\n')
        else:
            print('\nIDP fully-supervised\n')

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

        if self.mode == 'train' and random.uniform(0,1)>0.7:

            L = len(self.proteins[index])
            left = randint(0, L // 4)
            right = randint(L // 2, L)
            x = [self.w2i[amino] for amino in self.proteins[index]][left:right]
            if self.use_elmo:
                x = self.proteins[index][left:right]

            else:
                x = torch.LongTensor(x)
            y = [int(i) for i in self.annotations[index]][left:right]
            # x = torch.LongTensor(x)  # .unsqueeze(-1)
            y = torch.LongTensor(y)  # .unsqueeze(-1)
            # x = torch.nn.functional.one_hot(x, num_classes=20).float()

            # print(y.shape, y1.shape)
            # print(x[:5] , y1[:5])
            # print(x,y)
            if self.ssl:
                return x, x
            return x, y  # 1

        x = [self.w2i[amino] for amino in self.proteins[index]]
        y = [int(i) for i in self.annotations[index]]
        # print(len(x),len(y),len(self.proteins[index]),len(self.annotations[index]))
        if self.use_elmo:
            x = self.proteins[index]

        else:
            x = torch.LongTensor(x)
        y = torch.LongTensor([int(i) for i in self.annotations[index]])  # .unsqueeze(-1)
        # print(x.shape,y.shape)
        #        assert x.shape == y.shape, print(self.names[index])
        y1 = torch.nn.functional.one_hot(y, num_classes=2)
        # x = torch.nn.functional.one_hot(x, num_classes=20).float()
        # print(y,y1)
        # print(x,y)
        if self.ssl:
            return x, x
        return x, y  # 1

class Disorder723(Dataset):
    def __init__(self, config, mode):
        train_filepath = "data/idp_seq_2_seq/disorder723/train_723.txt"
        dev_filepath = "data/idp_seq_2_seq/disorder723/disorder723.txt"
        test_filepath = ""

        cwd = config.cwd
        if mode == train_prefix:
            self.names, self.annotations, self.proteins, self.classes, self.w2i = read_data_(
                os.path.join(cwd, train_filepath))
            self.augment = True
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, _, _ = read_data_(
                os.path.join(cwd, dev_filepath))
            self.mode = mode
            self.augment = False
        self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                        'V', 'W', 'X', 'Y']

        indixes = list(range(len(self.classes)))
        print(self.classes)
        self.w2i = dict(zip(self.classes, indixes))
        # print('classes\n\n', self.classes,len(self.classes))
        self.ssl = config.dataset.type == 'SSL'
        self.use_elmo = config.dataset.use_elmo
        if self.use_elmo:
            print('\n USE ELMO \n')
            # model_dir = Path('/config/uniref50_v2')
            # weights = model_dir / 'weights.hdf5'
            # options = model_dir / 'options.json'
            # self.embedder = ElmoEmbedder(options, weights, cuda_device=-1)
            if self.ssl:
                print('\n SELF-SUPERVISED\n')
            else:
                print('\nIDP fully-supervised\n')

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

        if self.mode == 'train' and self.augment:

            L = len(self.proteins[index])
            left = randint(0, L // 4)
            right = randint(L // 2, L)
            x = [self.w2i[amino] for amino in self.proteins[index]]  # [left:right]
            if self.use_elmo:
                x = self.proteins[index]
                # print(seq)
                # x =  torch.FloatTensor(self.embedder.embed_sentence(list(seq))).sum(dim=0).cpu()
            else:
                x = torch.LongTensor(x)
            y = [int(i) for i in self.annotations[index]]  # [left:right]
            # x = torch.LongTensor(x)  # .unsqueeze(-1)
            y = torch.LongTensor(y)  # .unsqueeze(-1)
            # x = torch.nn.functional.one_hot(x, num_classes=20).float()

            # print(y.shape, y1.shape)
            # print(x[:5] , y1[:5])
            # print(x,y)
            if self.ssl:
                return x, x
            return x, y  # 1

        x = [self.w2i[amino] for amino in self.proteins[index]]
        y = [int(i) for i in self.annotations[index]]
        # print(len(x),len(y),len(self.proteins[index]),len(self.annotations[index]))
        if self.use_elmo:
            seq = self.proteins[index]
            # print(seq)
            x = seq
        else:
            x = torch.LongTensor(x)
        y = torch.LongTensor([int(i) for i in self.annotations[index]])  # .unsqueeze(-1)
        # print(x.shape,y.shape)
#        assert x.shape == y.shape, print(self.names[index])
        y1 = torch.nn.functional.one_hot(y, num_classes=2)
        # x = torch.nn.functional.one_hot(x, num_classes=20).float()
        # print(y,y1)
        # print(x,y)
        if self.ssl:
            return x, x
        return x, y  # 1


class CAID2018_Disprot(Dataset):
    def __init__(self, config, mode):

        dev_filepath = "data/CAID_data_2018/disprot-binding-all.txt"
        test_filepath = ""

        cwd = config.cwd


        self.names, self.annotations, self.proteins, _, _ = read_data_(
            os.path.join(cwd, dev_filepath))
        self.mode = mode
        self.augment = False
        self.classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                        'V', 'W', 'X', 'Y']

        indixes = list(range(len(self.classes)))
        print(self.classes)
        self.w2i = dict(zip(self.classes, indixes))
        # print('classes\n\n', self.classes,len(self.classes))
        self.ssl = config.dataset.type == 'SSL'
        self.use_elmo = config.dataset.use_elmo
        if self.use_elmo:
            print('\n USE ELMO \n')
            # model_dir = Path('/config/uniref50_v2')
            # weights = model_dir / 'weights.hdf5'
            # options = model_dir / 'options.json'
            # self.embedder = ElmoEmbedder(options, weights, cuda_device=-1)
            if self.ssl:
                print('\n SELF-SUPERVISED\n')
            else:
                print('\nIDP fully-supervised\n')

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):


        x = [self.w2i[amino] for amino in self.proteins[index]]
        y = [int(i) for i in self.annotations[index]]
        # print(len(x),len(y),len(self.proteins[index]),len(self.annotations[index]))
        if self.use_elmo:
            seq = self.proteins[index]
            # print(seq)
            x = seq
        else:
            x = torch.LongTensor(x)
        y = torch.LongTensor([int(i) for i in self.annotations[index]])  # .unsqueeze(-1)

        return x, y  # 1