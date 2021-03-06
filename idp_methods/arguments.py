import argparse
import json
import re

import numpy as np
import sklearn.metrics

# paths to idp programs
cast = 'idp_methods/cast-linux/cast'
seg = 'idp_methods/ncbi-seg_0.0.20000620.orig/seg'
flps = 'idp_methods/fLPS/bin/linux/fLPS'
iupred2a = 'idp_methods/iupred2a/main.py'

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='cast', help='select method for detecting IDP regions',
                        choices=('cast', 'seg', 'iupred2a'))
    parser.add_argument('--log_interval', type=int, default=1000, help='steps to log.info metrics and loss')
    parser.add_argument('--dataset_name', type=str, default="COVIDx", help='dataset name COVIDx or COVID_CT')

    parser.add_argument('--tensorboard', action='store_true', default=True)

    parser.add_argument('--root_path', type=str, default='./data/data',
                        help='path to dataset ')
    parser.add_argument('--save', type=str, default='./results',
                        help='path to checkpoint save directory ')
    args = parser.parse_args()
    return args


def cast_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-help',
                        help='... print this text                 -thr t   ... set the threshold score for reported '
                             'regions default is 40' \
                             't should be an integer number' \
                             '-stat    ... outputs statistics information to file LICENSE.txt' \
                             '-matrix  ... use different mutation matrix (.mat) file' \
                             '-verbose ... verbose mode prints filtering information to standard output' \
                             '-stderr  ... verbose mode prints filtering information to standard error ')

    parser.add_argument('-thr', default=40, type=int,
                        help='... print this text                 -thr t   ... set the threshold score for reported ')
    parser.add_argument('-stat', default=True, type=bool, help='... outputs statistics information to file LICENSE.txt')
    parser.add_argument('-verbose', default=True, type=bool,
                        help='verbose mode prints filtering information to standard output')
    parser.add_argument('-stderr', default=True, type=bool,
                        help='verbose mode prints filtering information to standard error')
    args = parser.parse_args()
    return args


def seg_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', action='store_true', default=True,
                        help="-x  each input sequence is represented by a single output" \
                             "sequence with low-complexity regions replaced by" \
                             "strings of 'x' characters ")
    parser.add_argument('-c', type=int, default=60, help='-c <chars> number of sequence characters/line (default 60)')

    parser.add_argument('-m', type=int, default=0, help='<size> minimum length for a high-complexity segment'
                                                        '(default 0).  Shorter segments are merged with adjacent'
                                                        'low-complexity segments')
    parser.add_argument('-l', action='store_true', default=False,
                        help='  show only low-complexity segments (fasta format)')
    # parser.add_argument('-h',  action='store_true', default=False,
    #                     help='show only high-complexity segments (fasta format)')
    parser.add_argument('-a', action='store_true', default=False,
                        help='show all segments (fasta format)')
    parser.add_argument('-n', action='store_true', default=False,
                        help='do not add complexity information to the header line')
    parser.add_argument('-o', action='store_true', default=False,
                        help='show overlapping low-complexity segments (default merge)')

    parser.add_argument('-t', type=int, default=100, help='maximum trimming of raw segment (default 100)')
    parser.add_argument('-p', action='store_true', default=False,
                        help='prettyprint each segmented sequence (tree format)')
    parser.add_argument('-q', action='store_true', default=False,
                        help='prettyprint each segmented sequence (block format)')
    args = parser.parse_args()
    return args
    # a = "-x  each input sequence is represented by a single output" \
    #     "sequence with low-complexity regions replaced by" \
    #     "strings of 'x' characters "
    # #
    # # -c <chars> number of sequence characters/line (default 60)
    # # -m <size> minimum length for a high-complexity segment
    # #     (default 0).  Shorter segments are merged with adjacent
    # #     low-complexity segments
    # # -l  show only low-complexity segments (fasta format)
    # # -h  show only high-complexity segments (fasta format)
    # # -a  show all segments (fasta format)
    # # -n  do not add complexity information to the header line
    # # -o  show overlapping low-complexity segments (default merge)
    # # -t <maxtrim> maximum trimming of raw segment (default 100)
    # # -p  prettyprint each segmented sequence (tree format)
    # # -q  prettyprint each segmented sequence (block format) "


def select_method(method: str):
    if method == 'cast':
        cargs = cast_args()
        method_args_list = [cast]
        if cargs.verbose:
            method_args_list.append('-verbose')
        elif cargs.stderr:
            method_args_list.append('-stderr')
        if cargs.stat:
            method_args_list.append('-stat')

    elif method == 'seg':
        method_args_list = [seg]
        cargs = seg_args()

        # if cargs.x:
        #     method_args_list.append('-x')

    elif method == 'flps':
        method_args_list = [flps]
    elif method == 'iupred2a':
        method_args_list=['python',iupred2a,'short']
    return method_args_list

