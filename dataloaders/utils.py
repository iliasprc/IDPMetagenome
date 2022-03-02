import json

import numpy as np
import logging
from typing import Union, List, Tuple, Sequence, Dict, Any

import numpy as np
import torch

from dataloaders.tokenizer import TAPETokenizer

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


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def remove_blanks(path):
    with open(path, 'r') as f:
        data = f.read().splitlines()
    with open('/home/iliask/PycharmProjects/MScThesis/s.txt', 'w') as f1:
        proteins = []
        names = []
        annotations = []
        sequence = ''
        annot = ''
        for line in data:
            # print(i)
            # if ">" in i:
            # if "" in i:
            #     print(i)

            if line.rstrip():
                print(line)
                # print(line,'\n')
                if ">" in line:
                    print(f'{sequence}\n{annot}\n{line}')
                    f1.write(f'{sequence}\n{annot}\n{line}')
                    sequence = ''
                    annot = ''
                if not has_numbers(line):
                    # print(sequence)
                    annot = ''
                    sequence += line

                elif has_numbers(line):
                    annot += line
                    # print(sequence)
            # else:
            #     print('not rstrip',line)
        print(f'{sequence}\n{annot} ')


def create_annot_fasta(path):
    name = path.split('/')[-1]
    pathfolder, name = path.rsplit('/', 1)
    name_annot = f'{pathfolder}/annot_{name}'
    print(name, path, name_annot)
    name_prot = f'{pathfolder}/data_{name}'
    names, annotations, proteins, classes, w2i = read_data_(path)
    with open(name_annot, 'w') as f:
        for i in range(len(names)):
            f.write(f"{names[i]}\n{annotations[i]}\n")
    with open(name_prot, 'w') as f:
        for i in range(len(names)):
            f.write(f"{names[i]}\n{proteins[i]}\n")


def read_fidpnn_dataset(path):
    with open(path, 'r') as f:
        data = f.read().splitlines()
        proteins = data[::7]
        aminos = data[1::7]

        annot = data[2::7]

    return proteins, aminos, annot


def read_idp_dataset(path):
    with open(path, 'r') as f:
        data = f.read().splitlines()
        proteins_ids = data[::3]
        sequences = data[1::3]

        annotations = data[2::3]

    return proteins_ids, sequences, annotations


def read_data_(path):
    classes = []
    with open(path, 'r') as f:
        data = f.read().splitlines()

        proteins = []
        names = []
        annotations = []
        sequence = ''
        annot = ''
        cou = 0
        c = 0
        for idx, line in enumerate(data):

            if line.rstrip():
                #  print(line)
                if ">" in line:
                    names.append(line)
                    c += 1
                elif not has_numbers(line):
                    cou += 1
                    for chari in line:
                        if chari not in classes:
                            classes.append(chari)
                    proteins.append(line)
                elif has_numbers(line):
                    annotations.append(line)
        # print(len(names), len(proteins),len(annotations))
        assert len(names) == len(proteins), print(len(names), len(proteins))
        assert len(proteins) == len(annotations), print(len(annotations), len(proteins))
        classes = sorted(classes)
        indixes = list(range(len(classes)))
        # print(classes)
        w2i = dict(zip(classes, indixes))
        # print(w2i)
        # print(len(classes))
        return names, annotations, proteins, classes, w2i


def read_mobidb4_json(json_path):
    big_dict = {}
    with open(json_path, 'r') as f:

        for idx, line in enumerate(f):

            pred = line.strip('\n')
            d = json.loads(pred)
            print(d.keys())
            # print(pred)
            keys = d.keys()
            len = d['length']
            predictors = ['prediction-disorder-iupl', 'prediction-disorder-iups',
                          'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
                          'prediction-disorder-glo']
            # 'prediction-disorder-seg'
            # prediction_disorder_th_50 = d['prediction-disorder-th_50']
            # prediction_disorder_iupl = d['prediction-disorder-iupl']
            # prediction_disorder_iups = d['prediction-disorder-iups']
            # prediction_disorder_espN = d['prediction-disorder-espN']
            # prediction_disorder_espX = d['prediction-disorder-espX']
            # prediction_disorder_glo = d['prediction-disorder-glo']
            # regions_prediction_disorder_th_50 = d['prediction-disorder-th_50']['regions']
            # regions_prediction_disorder_iupl = d['prediction-disorder-iupl']['regions']
            # regions_prediction_disorder_iups = d['prediction-disorder-iups']['regions']
            # regions_prediction_disorder_espN = d['prediction-disorder-espN']['regions']
            # regions_prediction_disorder_espX = d['prediction-disorder-espX']['regions']
            # regions_prediction_disorder_glo = d['prediction-disorder-glo']['regions']
            # regions_th_50 = np.zeros(len)
            # regions_iupl = np.zeros(len)
            # regions_iups = np.zeros(len)
            # regions_espN = np.zeros(len)
            # regions_espX = np.zeros(len)
            # regions_espD = np.zeros(len)
            # regions_seg = np.zeros(len)
            # regions_glo = np.zeros(len)
            lca = {}

            for predictor in predictors:
                lca[predictor] = np.zeros(len)
                regions = d[predictor]['regions']
                # scores = d[predictor]['scores']
                if regions != 'None':
                    for area in regions:
                        start, end = area
                        lca[predictor][start:end] = 1
                # else:
                #     print('rdafgdsfdsf')
            # print(lca)
            # lca[predictor] = lca[predictor].tolist()

            big_dict[str(idx)] = lca

            # regions_prediction_disorder_dis465 = d['prediction-disorder-dis465']
            # prediction_disorder_disHL = d['prediction-disorder-disHL']

            # if 'prediction-disorder-th_50' not in keys:
            #     print('prediction-disorder-espX')

    # a_file = open('/home/iliask/PycharmProjects/MScThesis/results/mobidb/mxd494_val_pred.pkl', "w")
    # pickle.dump(big_dict, a_file)
    #
    # a_file.close()
    np.save('/home/iliask/PycharmProjects/MScThesis/results/mobidb/d723_test.npy', big_dict)


def convert_mobidb4_json(json_path, out_path='/home/iliask/PycharmProjects/MScThesis/results/mobidb/test_mobi_out.txt'):
    big_dict = {}
    fout = open(out_path, 'w')
    with open(json_path, 'r') as f:

        for idx, line in enumerate(f):

            pred = line.strip('\n')
            d = json.loads(pred)
            print(d.keys())
            fout.write(f'>sequence{idx}\n')
            # print(pred)
            keys = d.keys()
            len = d['length']
            predictors = ['prediction-disorder-iupl', 'prediction-disorder-iups',
                          'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
                          'prediction-disorder-glo']
            # 'prediction-disorder-seg'
            # prediction_disorder_th_50 = d['prediction-disorder-th_50']
            # prediction_disorder_iupl = d['prediction-disorder-iupl']
            # prediction_disorder_iups = d['prediction-disorder-iups']
            # prediction_disorder_espN = d['prediction-disorder-espN']
            # prediction_disorder_espX = d['prediction-disorder-espX']
            # prediction_disorder_glo = d['prediction-disorder-glo']
            # regions_prediction_disorder_th_50 = d['prediction-disorder-th_50']['regions']
            # regions_prediction_disorder_iupl = d['prediction-disorder-iupl']['regions']
            # regions_prediction_disorder_iups = d['prediction-disorder-iups']['regions']
            # regions_prediction_disorder_espN = d['prediction-disorder-espN']['regions']
            # regions_prediction_disorder_espX = d['prediction-disorder-espX']['regions']
            # regions_prediction_disorder_glo = d['prediction-disorder-glo']['regions']
            # regions_th_50 = np.zeros(len)
            # regions_iupl = np.zeros(len)
            # regions_iups = np.zeros(len)
            # regions_espN = np.zeros(len)
            # regions_espX = np.zeros(len)
            # regions_espD = np.zeros(len)
            # regions_seg = np.zeros(len)
            # regions_glo = np.zeros(len)
            lca = {}

            for predictor in predictors:
                lca[predictor] = np.zeros(len)
                regions = d[predictor]['regions']
                # scores = d[predictor]['scores']
                if regions != 'None':
                    for area in regions:
                        start, end = area
                        lca[predictor][start:end] = 1
                np_to_str = "".join([str(int(x)) for x in lca[predictor]])
                fout.write(f"{np_to_str}\n")
                # else:
                #     print('rdafgdsfdsf')
            # print(lca)
            # lca[predictor] = lca[predictor].tolist()

            big_dict[str(idx)] = lca

            # regions_prediction_disorder_dis465 = d['prediction-disorder-dis465']
            # prediction_disorder_disHL = d['prediction-disorder-disHL']

            # if 'prediction-disorder-th_50' not in keys:
            #     print('prediction-disorder-espX')

    # a_file = open('/home/iliask/PycharmProjects/MScThesis/results/mobidb/mxd494_val_pred.pkl', "w")
    # pickle.dump(big_dict, a_file)
    #
    # a_file.close()
    np.save('/home/iliask/PycharmProjects/MScThesis/results/mobidb/d723_test.npy', big_dict)


# convert_mobidb4_json('/home/iliask/PycharmProjects/MScThesis/results/mobidb/d723_test.json')
#
# train_mxd494 = np.load('/home/iliask/PycharmProjects/MScThesis/results/mobidb/mxd494_val.npy', allow_pickle=True)
# print(a.item().keys())
#

def mobi_db_annot():
    protein_ids, sequences, annotations = read_idp_dataset(
        '/data/disorder723/disorder723.txt')

    len_ = len(protein_ids)
    with open('/data/disorder723/data/test_d723.fasta', 'w') as f:
        for i in range(len_):
            f.write(f'{protein_ids[i]}\n')
            f.write(f'{sequences[i]}\n')
    f.close()
    with open('/data/disorder723/annotations/test_d723.fasta',
              'w') as f:
        for i in range(len_):
            f.write(f'{protein_ids[i]}\n')
            f.write(f'{annotations[i]}\n')


remove_blanks('/home/iliask/PycharmProjects/MScThesis/results/seg/mgm_simulationassembly.txt')
