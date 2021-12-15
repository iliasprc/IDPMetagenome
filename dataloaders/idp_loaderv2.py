import logging
from typing import Union, List, Tuple, Sequence, Dict, Any

import numpy as np
import torch

from .tokenizer import TAPETokenizer

train_prefix = "train"
dev_prefix = "val"
test_prefix = "test"
import os
from torch.utils.data import DataLoader, RandomSampler, Dataset
from dataloaders.utils._sampler import BucketBatchSampler
from dataloaders.utils.utils import read_data_, read_fidpnn_dataset

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
    def __init__(self, cwd: str, split: str, tokenizer: Union[str, TAPETokenizer] = 'iupac', ):
        train_filepath = "data/idp_seq_2_seq/train/all_train.txt"
        dev_filepath = "data/idp_seq_2_seq/validation/all_valid.txt"

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
        self.ignore_index = -1

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):

        item = self.proteins[index]
        token_ids = self.tokenizer.encode(item)  #
        # print(item,'\n',token_ids)
        input_mask = np.ones_like(token_ids)
        labels = np.asarray([int(i) for i in self.annotations[index]])
        #        assert token_ids.shape == labels.shape, print(self.names[index])
        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, self.ignore_index))

        output = {'input_ids' : input_ids,
                  'input_mask': input_mask,
                  'targets'   : ss_label}

        return output


class FIDPNN_idp_dataset(Dataset):
    def __init__(self, cwd: str, split: str, tokenizer: Union[str, TAPETokenizer] = 'iupac', ):
        train_filepath = "data/fidpnn_data/flDPnn_Training_Annotation.txt"
        dev_filepath = "data/fidpnn_data/flDPnn_Validation_Annotation.txt"
        test_filepath = "data/fidpnn_data/flDPnn_DissimiTest_Annotation.txt"

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if split == train_prefix:
            self.names, self.proteins, self.annotations = read_fidpnn_dataset(
                os.path.join(cwd, train_filepath))
            self.augment = True
            self.split = split
        elif split == dev_prefix:
            self.names, self.proteins, self.annotations = read_fidpnn_dataset(
                os.path.join(cwd, dev_filepath))
            self.split = split
            self.augment = False
        elif split == test_prefix:
            self.names, self.proteins, self.annotations = read_fidpnn_dataset(
                os.path.join(cwd, test_filepath))
            self.split = split
            self.augment = False
        self.classes = self.tokenizer.tokens
        self._num_examples = len(self.proteins)
        self.ignore_index = -1

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):

        item = self.proteins[index]
        token_ids = self.tokenizer.encode(item)  #
        # print(item,'\n',token_ids)
        input_mask = np.ones_like(token_ids)
        labels = np.asarray([int(i) for i in self.annotations[index]])
        #        assert token_ids.shape == labels.shape, print(self.names[index])
        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, self.ignore_index))

        output = {'input_ids' : input_ids,
                  'input_mask': input_mask,
                  'targets'   : ss_label}

        return output


class Disorder723(Dataset):
    def __init__(self, args, mode):
        train_filepath = "data/idp_seq_2_seq/disorder723/train_723.txt"
        dev_filepath = "data/idp_seq_2_seq/disorder723/disorder723.txt"
        test_filepath = ""

        cwd = args.cwd
        if mode == train_prefix:
            self.names, self.annotations, self.proteins, _, _ = read_data_(
                os.path.join(cwd, train_filepath))
            self.augment = True
            self.mode = mode
        elif mode == dev_prefix:
            self.names, self.annotations, self.proteins, _, _ = read_data_(
                os.path.join(cwd, dev_filepath))
            self.mode = mode
            self.augment = False

        self.classes = self.tokenizer.tokens
        self._num_examples = len(self.proteins)
        self.ignore_index = -1

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):

        item = self.proteins[index]
        token_ids = self.tokenizer.encode(item)  #
        # print(item,'\n',token_ids)
        input_mask = np.ones_like(token_ids)
        labels = np.asarray([int(i) for i in self.annotations[index]])
        #        assert token_ids.shape == labels.shape, print(self.names[index])
        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, self.ignore_index))

        output = {'input_ids' : input_ids,
                  'input_mask': input_mask,
                  'targets'   : ss_label}

        return output




class MXD494_idp_dataset(Dataset):
    def __init__(self, cwd: str, split: str, tokenizer: Union[str, TAPETokenizer] = 'iupac', ):
        train_filepath = "data/idp_seq_2_seq/mxd494/MXD494_train_all.txt"

        test_filepath = "data/idp_seq_2_seq/mxd494/MXD494.txt"
        dev_filepath = test_filepath

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
        self.ignore_index = -1

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):

        item = self.proteins[index]
        token_ids = self.tokenizer.encode(item)  #
        # print(item,'\n',token_ids)
        input_mask = np.ones_like(token_ids)
        labels = np.asarray([int(i) for i in self.annotations[index]])
        #        assert token_ids.shape == labels.shape, print(self.names[index])
        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, self.ignore_index))

        output = {'input_ids' : input_ids,
                  'input_mask': input_mask,
                  'targets'   : ss_label}

        return output


def setup_loader(dataset: Dataset,
                 batch_size: int,
                 local_rank: int,
                 n_gpu: int,
                 gradient_accumulation_steps: int,
                 num_workers: int) -> DataLoader:
    sampler = RandomSampler(dataset)
    # batch_size = get_effective_batch_size(
    #     batch_size, local_rank, n_gpu, gradient_accumulation_steps) * n_gpu
    # WARNING: this will fail if the primary sequence is not the first thing the dataset returns
    batch_sampler = BucketBatchSampler(
        sampler, batch_size, False, lambda x: len(x[0]), dataset)

    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,  # type: ignore
        batch_sampler=batch_sampler)

    return loader


def idp_dataset(args, cwd):
    tokenizer = TAPETokenizer(vocab='iupac')
    if args.dataset == 'DM':

        train_dataset = DM_idp_dataset(cwd=cwd, split='train', tokenizer=tokenizer)

        train_loader = setup_loader(train_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                    gradient_accumulation_steps=args.gradient_accumulation, num_workers=args.num_workers)
        val_dataset = DM_idp_dataset(cwd=cwd, split='val', tokenizer=tokenizer)

        val_loader = setup_loader(val_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                  gradient_accumulation_steps=args.gradient_accumulation,
                                  num_workers=2)
        return train_loader, val_loader, None, tokenizer.vocab
    elif args.dataset == 'MXD494':

        train_dataset = MXD494_idp_dataset(cwd=cwd, split='train', tokenizer=tokenizer)

        train_loader = setup_loader(train_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                    gradient_accumulation_steps=args.gradient_accumulation, num_workers=args.num_workers)
        val_dataset = MXD494_idp_dataset(cwd=cwd, split='val', tokenizer=tokenizer)

        val_loader = setup_loader(val_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                  gradient_accumulation_steps=args.gradient_accumulation,
                                  num_workers=2)
        return train_loader, val_loader, None, tokenizer.vocab
    elif args.dataset_name == 'FIDPNN':

        train_dataset = FIDPNN_idp_dataset(cwd=cwd, split='train', tokenizer=tokenizer)

        train_loader = setup_loader(train_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                    gradient_accumulation_steps=args.gradient_accumulation, num_workers=2)
        val_dataset = FIDPNN_idp_dataset(cwd=cwd, split='val', tokenizer=tokenizer)

        val_loader = setup_loader(val_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                  gradient_accumulation_steps=args.gradient_accumulation,
                                  num_workers=2)
        test_dataset = FIDPNN_idp_dataset(cwd=cwd, split='test', tokenizer=tokenizer)

        test_loader = setup_loader(test_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                   gradient_accumulation_steps=args.gradient_accumulation,
                                   num_workers=2)
        return train_loader, val_loader, test_loader, tokenizer.vocab
