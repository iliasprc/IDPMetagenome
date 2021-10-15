import argparse
import re
import json
import numpy as np
import sklearn.metrics

# paths to idp programs
cast = 'idp_programs/cast-linux/cast'
seg = 'idp_programs/ncbi-seg_0.0.20000620.orig/seg'
flps = 'idp_programs/fLPS/bin/linux/fLPS'


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
                             '-stat    ... outputs statistics information to file cast.stat' \
                             '-matrix  ... use different mutation matrix (.mat) file' \
                             '-verbose ... verbose mode prints filtering information to standard output' \
                             '-stderr  ... verbose mode prints filtering information to standard error ')

    parser.add_argument('-thr', default=40, type=int,
                        help='... print this text                 -thr t   ... set the threshold score for reported ')
    parser.add_argument('-stat', default=True, type=bool, help='... outputs statistics information to file cast.stat')
    parser.add_argument('-verbose', default=True, type=bool,
                        help='verbose mode prints filtering information to standard output')
    parser.add_argument('-stderr', default=True, type=bool,
                        help='verbose mode prints filtering information to standard error')
    args = parser.parse_args()
    return args


def seg_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=bool, action='store_true', default=False,
                        help="-x  each input sequence is represented by a single output" \
                             "sequence with low-complexity regions replaced by" \
                             "strings of 'x' characters ")
    parser.add_argument('-c', type=int, default=60, help='-c <chars> number of sequence characters/line (default 60)')

    parser.add_argument('-m', type=int, default=0, help='<size> minimum length for a high-complexity segment'
                                                        '(default 0).  Shorter segments are merged with adjacent'
                                                        'low-complexity segments')
    parser.add_argument('-l', type=bool, action='store_true', default=False,
                        help='  show only low-complexity segments (fasta format)')
    parser.add_argument('-h', type=bool, action='store_true', default=False,
                        help='show only high-complexity segments (fasta format)')
    parser.add_argument('-a', type=bool, action='store_true', default=False,
                        help='show all segments (fasta format)')
    parser.add_argument('-n', type=bool, action='store_true', default=False,
                        help='do not add complexity information to the header line')
    parser.add_argument('-o', type=bool, action='store_true', default=False,
                        help='show overlapping low-complexity segments (default merge)')

    parser.add_argument('-t', type=int, default=100, help='maximum trimming of raw segment (default 100)')
    parser.add_argument('-p', type=bool, action='store_true', default=False,
                        help='prettyprint each segmented sequence (tree format)')
    parser.add_argument('-q', type=bool, action='store_true', default=False,
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
    elif method == 'flps':
        method_args_list = [flps]
    return method_args_list


def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))


def post_process_seg_output(path):
    with open(path, 'r') as file1:
        data = file1.readlines()
        print(len(data))
        count = 0
        for idx, i in enumerate(data):

            if idx < 80000:
                i = i.strip()
                if '>disprot' in i:
                    continue
                if has_numbers(i):
                    # print(i)
                    if '-' in i[-5:]:
                        # print(i)
                        count += 1

            # print(i)
        print(count)
        # while True:
        #     count += 1
        #
        #     # Get next line from file
        #     line = file1.readline()
        #
        #     # if line is empty
        #     # end of file is reached
        #     if not line:
        #         print('break')
        #         break
        #     print("Line{}: {}".format(count, line.strip()))

        file1.close()


def post_process_cast_outputv1(path):
    with open(path, 'r') as file1:
        data = file1.readlines()
        print(len(data))
        count = 0
        for idx, i in enumerate(data):
            print(f'{i.strip()}----------')
            i = i.strip()
            if 'region' in i:
                count += 1
                # print(i)

            # if idx < 80000:
            #     i = i.strip()
            #     if '>disprot' in i:
            #         continue
            #     if has_numbers(i):
            #        # print(i)
            #         if '-' in i[-5:]:
            #             print(i)
            #             count+=1
            #
            #
            # # print(i)
        print(count)
        # while True:
        #     count += 1
        #
        #     # Get next line from file
        #     line = file1.readline()
        #
        #     # if line is empty
        #     # end of file is reached
        #     if not line:
        #         print('break')
        #         break
        #     print("Line{}: {}".format(count, line.strip()))

        file1.close()
        return count


def post_process_cast_output(path, ground_truth_path):
    with open(path, 'r') as file1:
        data = file1.readlines()
    file1.close()
    print(len(data))
    count = 0
    proteins = []
    predictions = []
    s = ''
    for idx, i in enumerate(data):

        i = i.strip('\n')
        # print(f'{i}')
        if 'region' in i:
            count += 1
            # print(i)
        if '>' in i:

            print(s)
            print(i)

            proteins.append(i)
            if s != '':
                predictions.append(s)
            s = ''
        else:
            s += i
    print(s)
    if s != '':
        predictions.append(s)
    print(len(proteins), len(predictions))
    idppreds = []
    for i in range(len(predictions)):
        print(predictions[i])
        s = predictions[i]
        s = s.replace('X', '1')
        s = re.sub('\D', '0', s)
        print(s)
        idppreds.append(s)
    print(len(idppreds))
    annotations = []
    with open(ground_truth_path, 'r') as file1:
        gt = file1.read().splitlines()
        print(gt)
        for i in gt:
            if not '>' in i:
                annotations.append(i)

    assert len(annotations) == len(idppreds)
    avgf1 = 0
    avg_mcc = 0
    for i in range(len(idppreds)):
        pred = [int(c) for c in idppreds[i]]
        target = [int(c) for c in annotations[i]]
        #print(len(pred), len(target))
        assert len(pred) == len(target)
        pred = np.array(pred)
        target = np.array(target)
        auc = sklearn.metrics.accuracy_score(target, pred)
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(target, pred)
        f1 = sklearn.metrics.f1_score(target, pred, average='macro')
        mcc = sklearn.metrics.matthews_corrcoef(np.where(target < 1, -1, 1), np.where(pred < 1, -1, 1))
        # mcc = sklearn.metrics.matthews_corrcoef(target,pred)
        # print(np.where(target<1,target,-1),target)
       # print(precision, f1, mcc)
        avg_mcc += mcc
        avgf1 += f1
        confusion_matrix = sklearn.metrics.confusion_matrix(target, pred)
        #
        # FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        # FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        # TP = np.diag(confusion_matrix)

        cm = confusion_matrix.ravel()
        TN, FP, FN, TP = cm
       # print(cm, cm.sum(), len(pred))

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        print(ACC)

        #print(auc)
    print(avgf1 / len(idppreds), avg_mcc / len(idppreds))
    return count


def read_caid_data(path):
    assert path[-4:] == '.txt', print(f"NOT txt file")
    protein_ids = []
    proteins = []
    annotations = []
    with open(path, 'r') as f:
        data = f.read().splitlines()
        # print(data)

        for i in data:
            # print(i)
            if '>' in i:
                protein_ids.append(i)
            elif (('0' not in i) and ('1' not in i)):
                proteins.append(i)
            else:
                annotations.append(i)
        assert len(proteins) == len(protein_ids) == len(annotations), f'error in reading txt file with proteins'
        print(len(proteins), len(protein_ids), len(annotations))
        print(path)
    f.close()
    print(path.rsplit('/', 1))
    path, name = path.rsplit('/', 1)
    data_path = f'/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/data/CAID_data_2018/fasta_files/data_{name}'
    annot_path = f'/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/data/CAID_data_2018/annotation_files/annot_{name}'
    with open(data_path, 'w') as f:
        for i in range(len(proteins)):
            f.write(f'{protein_ids[i]}\n{proteins[i]}\n')
    f.close()
    with open(annot_path, 'w') as f:
        for i in range(len(proteins)):
            f.write(f'{protein_ids[i]}\n{annotations[i]}\n')
    f.close()


# def create_caid_fasta_file(proteins,protein_ids,annotations):


def read_json(path):
    # Opening JSON file
    with open(path, 'r') as f:
        data = json.load(f)
        print(data.keys())
        size = data['size']
        data = data['data']
        print(data[0])
        print(size)


# read_json('/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/data/DisProt release_2021_08.json')
# import glob
# files_ = sorted(glob.glob(f'/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/data/CAID_data_2018/**.txt'))
# print(files_)
# for i in files_:
#
#     read_caid_data(i)

post_process_cast_output('/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/results/cast/data_disprot-disorder.out.txt',
                         '/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/data/CAID_data_2018/annotation_files/annot_disprot-disorder.txt')
