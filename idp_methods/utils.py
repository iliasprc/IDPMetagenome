import json
import re

import numpy as np
import sklearn.metrics


def read_fldpnn_file(path):
    step = 12
    with open(path, 'r') as f:
        data = f.read().replace(',', '').splitlines()
        proteins_ids = data[::step]
        sequences = data[1::step]
        #
        predictions = data[2::step]

    # for i in range(len(proteins_ids)):
    #     print(f'{proteins_ids[i]}')
    #     print(f'{predictions[i]}')

    # print(proteins_ids)
    # print(sequences)
    # print(predictions)
    print(len(proteins_ids), len(sequences), len(predictions))
    return proteins_ids, sequences, predictions


def read_swissprot(path):
    annotations = []
    with open(path, 'r') as file1:
        gt = file1.read().splitlines()
        # print(gt)
    s = ''
    for i in gt:
        print(i)
        if not '>' in i:
            s += re.sub('\D', '1', i)

        else:
            annotations.append(s)
            s = ''
    annotations.pop(0)
    annotations.append(s)
    print(len(annotations), (annotations[-1]))
    return annotations


def read_disprot(path):
    annotations = []
    with open(path, 'r') as file1:
        gt = file1.read().splitlines()
        # print(gt)
    s = ''
    for i in gt:
        # print(i)
        if not '>' in i:
            s += re.sub('\D', '1', i)

        else:
            annotations.append(s)
            s = ''
    annotations.pop(0)
    annotations.append(s)
    # print(len(annotations), (annotations[-1]))
    return annotations


def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))


def post_process_seg_output(path):
    with open(path, 'r') as file1:
        data = file1.readlines()
        print(len(data))
        count = 0
        protein_count = 0

        protein_seq = ''
        idpr = ''
        for idx, i in enumerate(data):

            # i = i.strip()
            # print(i[:36])
            # print(i[30:40])
            if i.rstrip():

                if '>' in i.strip():
                    protein_count += 1
                    # continue
                elif has_numbers(i.strip()):
                    # if String contains numbers

                    # check output of seg if number are on left or right region
                    # and find IDRs
                    #
                    # print(i)
                    # print(i[30:40].strip())
                    ids = [int(x) for x in i[30:40].strip().split('-')]
                    # print(ids)
                    # print(i.strip())
                    # print(len(i))
                    # print(i)
                    if '-' in i.strip()[-5:]:
                        # print(i)
                        count += 1
                # else:
                #     print(i.strip(),has_numbers(i.strip()))
            # else:
            #     print(i)
            # print(i)
        print('Number of IDRs ', count)
        print('Number of  Proteins ', protein_count)

    file1.close()
    return count


def seg_predictions(path, annotations=None):
    """

    Args:
        path: (str) path of prediction file
        annotations:

    Returns:

    """
    with open(path, 'r') as file1:
        data = file1.readlines()

        protein_count = 0

        predictions = []
        proteins = []
        pred = ''
        for idx, i in enumerate(data):

            i = i.strip()

            if '>' in i.strip():
                proteins.append(i.strip())
                #
                predictions.append(pred)
                pred = ''
                protein_count += 1

            else:

                region = i
                region = region.replace('x', '1')
                region = re.sub('\D', '0', region)

                pred += region

    predictions.append(pred)
    predictions.pop(0)

    return predictions, proteins


def read_mobidb4_json(json_path, out_path):
    """
    Read mobi-db-lite output in json format
    Args:
        json_path:
        out_path:
    """
    big_dict = {}
    with open(json_path, 'r') as f:

        for idx, line in enumerate(f):

            pred = line.strip('\n')
            d = json.loads(pred)
            # print(d.keys())
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
                if regions != 'None':
                    for area in regions:
                        start, end = area
                        lca[predictor][start:end] = 1
                # else:
                #     print('rdafgdsfdsf')
            # print(lca)
            # lca[predictor] = lca[predictor].tolist()

            big_dict[str(idx)] = lca

    np.save(out_path, big_dict)


def read_annotation_file(annotation_file_path):
    """
    Read annotation file that is in format

    >protein1
    00000111110111010000..........................

    Args:
        annotation_file_path:

    Returns: annotations

    """
    annotations = []
    with open(annotation_file_path, 'r') as file1:
        gt = file1.read().splitlines()
        # print(gt)
        for i in gt:
            if not '>' in i:
                annotations.append(i)
    return annotations


def cast_metrics_V2(path, ground_truth_path):
    with open(path, 'r') as file1:
        data = file1.readlines()
        # print(len(data))
        count = 0
        protein_seq = ''
        protein_count = 0
        protein_ids = []
        proteins = []
        s = ''
        idpr = ''
        idp_regions = []
        for idx, i in enumerate(data):
            # print(f'{i.strip()}')
            i = i.strip()

            if 'region' in i:
                count += 1

                # print(i)
                # print(re.sub('\D', '_', i))
                region_start = int(i[18:].split('to')[0].replace(" ", ""))
                region_end = int(i.split('to')[-1].split('corrected')[0].replace(" ", ""))
                # print(i[18:],region_start)
                idpr += f"{region_start},{region_end},"
                # print(f"{region_start},{region_end},")
            elif '>' in i:
                regions = []
                protein_ids.insert(protein_count, i)
                proteins.insert(protein_count, protein_seq)
                idp_regions.insert(protein_count, idpr)
                protein_count += 1
                # print(i, protein_seq)
                protein_seq = ''
                idpr = ''
            else:
                protein_seq += i
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
        # print(count)
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
        proteins.append(protein_seq)
        proteins.pop(0)
        idp_regions.append(idpr)
        idp_regions.pop(0)
        # print(len(proteins), len(protein_ids), len(idp_regions))
        # print(protein_ids[-1], '\n', proteins[-1], '\n', idp_regions[-1])

        file1.close()
        with open(path + 'postprocessed.txt', 'w') as f:
            for i in range(len(proteins)):
                f.write(f"{protein_ids[i]}\n{proteins[i]}\n{idp_regions[i]}\n")
        f.close()
        predictions = []
        for i in range(len(proteins)):
            s = (f"{protein_ids[i]}\n{proteins[i]}\n{idp_regions[i]}\n")
            sequence_len = len(proteins[i])
            # print(idp_regions[i].split(','))
            regions = idp_regions[i].split(',')
            if len(regions) == 1:
                pred = np.zeros(sequence_len)
            else:
                pred = np.zeros(sequence_len)
                regions = regions[:-1]
                regions = [int(i) for i in regions]
                # print(regions)
                regions = iter(regions)
                for x in regions:
                    start, end = x, next(regions)
                    pred[start:end] = 1
                # print(sequence_len, pred)
            pred_string = ''
            pred = pred.tolist()
            for i in pred:
                pred_string += str(int(i))
            # print(pred_string)
            predictions.append(pred_string)

            # print(regions)


def dataset_metrics(dataset_preds: np.array, dataset_target: np.array):
    pred = dataset_preds
    target = dataset_target
    # print(pred)
    # print(pred.shape,target.shape)
    confusion_matrix = sklearn.metrics.confusion_matrix(target, pred)
    # print(confusion_matrix)
    cm = confusion_matrix.ravel()

    TN, FP, FN, TP = cm

    # print(TN, FP, FN, TP)
    # print(cm, cm.sum(), len(pred))

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    FOR = 1.0 - NPV
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    F1 = TP / (TP + 0.5 * (FP + FN))
    # Overall accuracy

    BAC = (TPR + TNR) / 2.0

    # denominator = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    MCC = (PPV * TPR * TNR * NPV) ** 0.5 - ((FDR * FNR * FPR * FOR) ** 0.5)
    # print(FPR,TPR)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(target, pred)
    auc = sklearn.metrics.auc(fpr, tpr)
    s = f'\nTP\tTN\tFP\tFN\t\n{TP:.2f}\t{TN:.2f}\t{FP:.2f}\t{FN:.2f}\n' \
        f'F1  \t  MCC  \t  TPR  \t  TNR  \t  PPV  \t  NPV  \t  FPR  \t  FNR  \t  BAC\t AUC\t\n' \
        f'{F1 :.4f}\t{MCC :.4f}\t{TPR:.4f}\t{TNR:.4f}\t{PPV:.4f}\t{NPV:.4f}\t{FPR:.4f}\t{FNR:.4f}\t{BAC:.4f}\t{auc:.4f}'
    metr_dict = {'F1' : F1, 'MCC': MCC, 'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 'NPV': NPV, 'FPR': FPR, 'FNR': FNR,
                 'BAC': BAC, 'AUC': auc}
    return s, metr_dict


def target_metrics(idppreds, annotations):
    assert len(annotations) == len(idppreds), print(f'{len(annotations)}  != {len(idppreds)}')
    print(len(annotations))
    avgf1 = 0
    avg_mcc = 0
    avg_cm = [0, 0, 0, 0]
    dataset_preds = []
    dataset_target = []
    # print(idppreds)
    for i in range(len(idppreds)):
        # print(idppreds[i])
        pred = [int(c) for c in idppreds[i]]
        target = [int(c) for c in annotations[i]]
        # print(i, len(pred), len(target))
        assert len(pred) == len(target)
        dataset_preds += pred
        dataset_target += target

    pred = np.array(dataset_preds)
    target = np.array(dataset_target)

    confusion_matrix = sklearn.metrics.confusion_matrix(target, pred)

    cm = confusion_matrix.ravel()
    # print(cm)

    TN, FP, FN, TP = cm
    avg_cm[0] += TP  # / len(pred)
    avg_cm[1] += TN  # / len(pred)
    avg_cm[2] += FP  # / len(pred)
    avg_cm[3] += FN  # / len(pred)
    # print(TN, FP, FN, TP)
    # print(cm, cm.sum(), len(pred))

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    FOR = 1.0 - NPV
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    F1 = TP / (TP + 0.5 * (FP + FN))
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    BAC = (TPR + TNR) / 2.0
    numerotr = TP * TN - FP * FN
    # denominator = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    MCC = (PPV * TPR * TNR * NPV) ** 0.5 - ((FDR * FNR * FPR * FOR) ** 0.5)
    # MCC = numerotr/denominator

    s = f'{"TP":4}\t{"TN":4}\t{"FP":4}\t{"FN":4}\t{"F1":4}\t{"MCC":4}\t{"TPR":4}\t{"TNR":4}' \
        f'\t{"PPV":4}\t{"NPV":4}\t{"FPR":4}\t{"FNR":4}\t{"BAC":4}'
    print(s)
    s = f'{TP:d}\t{TN:d}\t{FP:d}\t{FN:d}\t{F1:.4f}\t{MCC:.4f}\t{TPR:.4f}\t{TNR:.4f}' \
        f'\t{PPV:.4f}\t{NPV:.4f}\t{FPR:.4f}\t{FNR:.4f}\t{BAC:.4f}'
    # print(auc)
    print(s)
    s = f'TP={TP:.2f}\tTN={TN:.2f}\tFP={FP:.2f}\tFN={FN:.2f}\tF1={F1 :.4f}\t   MCC {MCC :.4f}\n TPR {TPR:.4f} TNR ' \
        f'{TNR:.4f}  PPV {PPV:.4f}\nNPV {NPV:.4f} FPR {FPR:.4f} FNR {FNR:.4f} BAC {BAC:.4f}'
    # print(s)

    return s


def cast_metrics(predictions_path, ground_truth_path=None):
    with open(predictions_path, 'r') as file1:
        data = file1.readlines()
    file1.close()

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
        elif '>' in i:

            proteins.append(i)
            if s != '':
                predictions.append(s)
            s = ''
        else:
            s += i
    print('IDR regions from CAST ',count)
    if s != '':
        predictions.append(s)
    # print(len(proteins), len(predictions))
    idppreds = []
    for i in range(len(predictions)):
        # print(predictions[i])
        s = predictions[i]
        s = s.replace('X', '1')
        s = re.sub('\D', '0', s)
        # print(s)
        idppreds.append(s)

    return idppreds


def read_caid_data(path):
    """
    Function to read CAID 2018 data
    Args:
        path:
    """
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
    data_path = f'./data/CAID_data_2018/fasta_files/data_{name}'
    annot_path = f'./data/CAID_data_2018/annotation_files/annot_{name}'
    with open(data_path, 'w') as f:
        for i in range(len(proteins)):
            f.write(f'{protein_ids[i]}\n{proteins[i]}\n')
    f.close()
    with open(annot_path, 'w') as f:
        for i in range(len(proteins)):
            f.write(f'{protein_ids[i]}\n{annotations[i]}\n')
    f.close()


def create_annot_fasta(path, predictions, proteins):
    with open(path, 'w') as f:
        for i in range(len(proteins)):
            f.write(f"{proteins[i]}\n{predictions[i]}\n")


def read_json(path):
    # Opening JSON file
    with open(path, 'r') as f:
        data = json.load(f)
        print(data.keys())
        size = data['size']
        data = data['data']
        print(data[0])
        print(size)


def post_process_cast_output(path):
    with open(path, 'r') as file1:
        data = file1.readlines()
        print(len(data))
        count = 0
        for idx, i in enumerate(data):
            print(f'{i.strip()}')
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
        print('Number of IDRs', count)

        file1.close()
        return count


def read_mobidb4_json_full(json_path):
    big_dict = {}

    ordered = 0.0
    disordered = 0.0
    total_amino = 0.0
    ratio = 0.0
    num_regions = 0.0
    predictors = ['prediction-disorder-iupl', 'prediction-disorder-iups',
                  'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
                  'prediction-disorder-glo']
    regions_predictors = {'prediction-disorder-iupl':0, 'prediction-disorder-iups':0,
                  'prediction-disorder-espN':0, 'prediction-disorder-espX':0, 'prediction-disorder-espD':0,
                  'prediction-disorder-glo':0}
    fraction_predictors = {'prediction-disorder-iupl':0, 'prediction-disorder-iups':0,
                  'prediction-disorder-espN':0, 'prediction-disorder-espX':0, 'prediction-disorder-espD':0,
                  'prediction-disorder-glo':0}
    with open(json_path, 'r') as f:

        for idx, line in enumerate(f):

            pred = line.strip('\n')
            d = json.loads(pred)
            #print(d.keys())
            # print(pred)
            keys = d.keys()
            len = d['length']

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
            total_amino+=len
            for predictor in predictors:
                lca[predictor] = np.zeros(len)
                if predictor in keys:
                    regions = d[predictor]['regions']
                    disordered_fraction = d[predictor]['content_fraction']
                    fraction_predictors[predictor]+=disordered_fraction
                    # scores = d[predictor]['scores']
                    if regions != 'None':
                        for area in regions:
                            start, end = area
                            lca[predictor][start:end] = 1

                            regions_predictors[predictor]+=1
                else:
                    print(f'Predictor {predictor} not found')
        print(idx)
        for predictor in predictors:


            fraction_predictors[predictor] = fraction_predictors[predictor]/idx
    print(regions_predictors)
    print(fraction_predictors)
    for k,v in regions_predictors.items():
        print(f'{k}|{v}')
    for k,v in fraction_predictors.items():
        print(f'{k}|{v}')

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

# post_process_seg_output('/home/iliask/PycharmProjects/MScThesis/results/seg/mgm_simulationassembly.txt')
