import logging
import logging
import random
from copy import copy
from typing import Union, List, Tuple, Dict, Any

import numpy as np
import torch

from .tokenizer import TAPETokenizer

train_prefix = "train"
dev_prefix = "val"
test_prefix = "test"
import os
from torch.utils.data import DataLoader, RandomSampler, Dataset
from dataloaders.utils._sampler import BucketBatchSampler, pad_sequences
from dataloaders.utils.utils import read_data_, read_fidpnn_dataset

logger = logging.getLogger(__name__)

class LM_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, self.ignore_index))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output


class DM_LM(LM_Dataset):
    def __init__(self, cwd: str, split: str, tokenizer: Union[str, TAPETokenizer] = 'iupac', ):
        super(DM_LM, self).__init__()
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
        tokens = self.tokenizer.tokenize(item)  #
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)
        labels = np.asarray([int(i) for i in self.annotations[index]])
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        return masked_token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, self.ignore_index))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output


class FIDPNN_LM(LM_Dataset):
    def __init__(self, cwd: str, split: str, tokenizer: Union[str, TAPETokenizer] = 'iupac', ):
        super(FIDPNN_LM, self).__init__()
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

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output


class Disorder723LM(LM_Dataset):
    def __init__(self, args, mode):
        super(Disorder723LM, self).__init__()
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
        tokens = self.tokenizer.tokenize(item)  #
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)
        labels = np.asarray([int(i) for i in self.annotations[index]])
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        return masked_token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, self.ignore_index))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output


class MXD494_LM(LM_Dataset):
    def __init__(self, cwd: str, split: str, tokenizer: Union[str, TAPETokenizer] = 'iupac', ):
        super(MXD494_LM, self).__init__()
        train_filepath = "data/mxd494/MXD494_train_all.txt"

        test_filepath = "data/mxd494/MXD494.txt"
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
        tokens = self.tokenizer.tokenize(item)  #
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)
        labels = np.asarray([int(i) for i in self.annotations[index]])
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        return masked_token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, self.ignore_index))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                pass

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels


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

        train_dataset = DM_LM(cwd=cwd, split='train', tokenizer=tokenizer)

        train_loader = setup_loader(train_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                    gradient_accumulation_steps=args.gradient_accumulation,
                                    num_workers=args.num_workers)
        val_dataset = DM_LM(cwd=cwd, split='val', tokenizer=tokenizer)

        val_loader = setup_loader(val_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                  gradient_accumulation_steps=args.gradient_accumulation,
                                  num_workers=2)
        return train_loader, val_loader, None, tokenizer.vocab
    elif args.dataset == 'MXD494':

        train_dataset = MXD494_LM(cwd=cwd, split='train', tokenizer=tokenizer)

        train_loader = setup_loader(train_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                    gradient_accumulation_steps=args.gradient_accumulation,
                                    num_workers=args.num_workers)
        val_dataset = MXD494_LM(cwd=cwd, split='val', tokenizer=tokenizer)

        val_loader = setup_loader(val_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                  gradient_accumulation_steps=args.gradient_accumulation,
                                  num_workers=2)
        return train_loader, val_loader, None, tokenizer.vocab
    elif args.dataset_name == 'FIDPNN':

        train_dataset = FIDPNN_LM(cwd=cwd, split='train', tokenizer=tokenizer)

        train_loader = setup_loader(train_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                    gradient_accumulation_steps=args.gradient_accumulation, num_workers=2)
        val_dataset = FIDPNN_LM(cwd=cwd, split='val', tokenizer=tokenizer)

        val_loader = setup_loader(val_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                  gradient_accumulation_steps=args.gradient_accumulation,
                                  num_workers=2)
        test_dataset = FIDPNN_LM(cwd=cwd, split='test', tokenizer=tokenizer)

        test_loader = setup_loader(test_dataset, batch_size=args.batch_size, local_rank=-1, n_gpu=1,
                                   gradient_accumulation_steps=args.gradient_accumulation,
                                   num_workers=2)
        return train_loader, val_loader, test_loader, tokenizer.vocab
