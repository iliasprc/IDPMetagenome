
import json
import re

import numpy as np
import sklearn.metrics





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
        #print(i)
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
            if '>' in i.strip():
                protein_count += 1
                # continue
            elif has_numbers(i.strip()):
                #
                # print(i)
                print(i[30:40].strip())
                ids = [int(x) for x in i[30:40].strip().split('-')]
                print(ids)
                # print(i.strip())
                # print(len(i))
                # print(i)
                if '-' in i.strip()[-5:]:
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
    return count


def metrics_seg(path, annotations):
    with open(path, 'r') as file1:
        data = file1.readlines()
        #print(len(data))
        count = 0
        protein_count = 0

        protein_seq = ''
        idpr = ''
        predictions = []
        pred = ''
        for idx, i in enumerate(data):

            i = i.strip()
            #print(i)
            # i = i.strip()
            # print(i[:36])
            # print(i[30:40])
            if '>' in i.strip():
                # print(pred)
                predictions.append(pred)
                pred = ''
                protein_count += 1

                # continue
            else:
                #
                # print(i)
                # print(i.strip())
                region = i
                region = region.replace('x', '1')
                region = re.sub('\D', '0', region)
                #print(pred)
                pred += region

            # print(i)
        #print(count)
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
    predictions.append(pred)
    predictions.pop(0)
   # print(protein_count)
    #print(len(predictions))
    file1.close()
    # annotations = []
    # with open(ground_truth_path, 'r') as file1:
    #     gt = file1.read().splitlines()
    #     #print(gt)
    #     for i in gt:
    #         if not '>' in i:
    #             annotations.append(i)

    assert len(annotations) == len(predictions), print(len(annotations), len(predictions))
    target_metrics(predictions, annotations)
    return count

    return count


def post_process_iupred2a_out1(path, ground_truth_path):
    with open(path, 'r') as file1:
        data = file1.read().splitlines()
        print(len(data))
        count = 0
        pred = ''
        idppreds = []
        for idx, i in enumerate(data):
            # print({i.strip()})
            if '>' not in i:
                s = 0
                print(i.split('\t'))
                amino_score = i.split('\t')[-1]
                pred += amino_score


            else:

                idppreds.append(pred)
                pred = ''
                print(i)
    annotations = []
    idppreds.pop(0)
    idppreds.append(pred)
    with open(ground_truth_path, 'r') as file1:
        gt = file1.read().splitlines()
        # print(gt)
        for i in gt:
            if not '>' in i:
                annotations.append(i)

    assert len(annotations) == len(idppreds), print(f'{len(annotations)}  != {len(idppreds)}')
    print(len(annotations))
    avgf1 = 0
    avg_mcc = 0
    avg_cm = [0, 0, 0, 0]
    dataset_preds = []
    dataset_target = []
    TPR = 0
    # Specificity or true negative rate
    TNR = 0
    # Precision or positive predictive value
    PPV = 0
    # Negative predictive value
    NPV = 0
    # Fall out or false positive rate
    FPR = 0
    # False negative rate
    FNR = 0
    # False discovery rate
    FDR = 0
    F1 = 0
    # Overall accuracy
    ACC = 0

    BAC = 0

    for i in range(len(idppreds)):
        # print(idppreds[i])
        pred = [int(c) for c in idppreds[i]]
        target = [int(c) for c in annotations[i]]
        # print(i,len(pred), len(target))
        assert len(pred) == len(target)
        pred = np.array(pred)
        target = np.array(target)

        # mcc = sklearn.metrics.matthews_corrcoef(target,pred)
        # print(np.where(target<1,target,-1),target)
        # print(precision, f1, mcc)
        # avg_mcc += mcc

        # print(target,pred)
        confusion_matrix = sklearn.metrics.confusion_matrix(target, pred)

        cm = confusion_matrix.ravel()
        # print(cm)
        if 0 in cm:
            print(cm)
        if (len(cm) != 4):
            print('GAMWWWWWWWWWWWWWWWWWWWWW\n\n\n\n\n\n\n\n\n')
            TN = cm[0] - 3
            FP, FN, TP = 1, 1, 1
        else:
            TN, FP, FN, TP = cm

            # F1 = TP / (TP + 0.5 * (FP + FN))
        # avgf1 += F1

        # Sensitivity, hit rate, recall, or true positive rate
        TPR += TP / (TP + FN)
        # Specificity or true negative rate
        TNR += TN / (TN + FP)
        # Precision or positive predictive value
        PPV += TP / (TP + FP)
        # Negative predictive value
        NPV += TN / (TN + FN)
        # Fall out or false positive rate
        FPR += FP / (FP + TN)
        # False negative rate
        FNR += FN / (TP + FN)
        # False discovery rate
        FDR += FP / (TP + FP)
        F1 += TP / (TP + 0.5 * (FP + FN))
        # Overall accuracy
        ACC += (TP + TN) / (TP + FP + FN + TN)

        BAC += (TPR + TNR) / 2.0

        avg_cm[0] += TP
        avg_cm[1] += TN
        avg_cm[2] += FP
        avg_cm[3] += FN
        # print(TN, FP, FN, TP)
        # print(cm, cm.sum(), len(pred))

    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    # F1 = TP / (TP + 0.5 * (FP + FN))
    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    #
    # BAC = (TPR + TNR) / 2.0
    # print(auc)
    print(avgf1 / len(idppreds), avg_mcc / len(idppreds))
    avg_cm[0] = avg_cm[0]  # * len(predictions)
    avg_cm[1] = avg_cm[1]  # * len(predictions)
    avg_cm[2] = avg_cm[2]  # * len(predictions)
    avg_cm[3] = avg_cm[3]  # * len(predictions)
    print(avg_cm)
    print(
        f'TP,TN,FP,FN\n{avg_cm[0] / len(idppreds):.2f},{avg_cm[1] / len(idppreds):.2f},'
        f'{avg_cm[2] / len(idppreds):.2f},{avg_cm[3] / len(idppreds):.2f}\n F1 {avgf1 :.4f} F1 {F1 / len(idppreds)}  '
        f'MCC {avg_mcc :.4f}')
    print(
        f" TPR {TPR / len(idppreds):.4f} TNR {TNR / len(idppreds):.4f}  PPV {PPV / len(idppreds):.4f}\nNPV "
        f"{NPV / len(idppreds):.4f} FPR {FPR / len(idppreds):.4f} FNR {FNR / len(idppreds):.4f} BAC "
        f"{BAC / len(idppreds):.4f}")
    return count
    return count


def iupred2a_metrics(path, annotations):
    with open(path, 'r') as file1:
        data = file1.read().splitlines()
        print(len(data))
        count = 0
        pred = ''
        idppreds = []
        for idx, i in enumerate(data):
            # print({i.strip()})
            if '>' not in i:
                s = 0
                #print(i.split('\t'))
                amino_score = i.split('\t')[-1]
                pred += amino_score


            else:

                idppreds.append(pred)
                pred = ''
                #print(i)
    #annotations = []
    idppreds.pop(0)
    idppreds.append(pred)
    # with open(ground_truth_path, 'r') as file1:
    #     gt = file1.read().splitlines()
    #     # print(gt)
    #     for i in gt:
    #         if not '>' in i:
    #             annotations.append(i)

    assert len(annotations) == len(idppreds), print(f'{len(annotations)}  != {len(idppreds)}')

    target_metrics(idppreds, annotations)


# ,fn,fp,tn,tp
# D003_IUPred2A-long.out,19649,78275,203716,34955
# ,,bac,  csi, f05,  f1s,  f2s,  fnr,  fom,  fpr,  inf   ,mcc,mk,npv,ppv,tnr,tpr,thr
# 0.684,0.262,0.339,0.416,0.537,0.333,0.084,0.299,0.368,0.283,0.218,0.916,0.302,0.701,0.667,0.477

def post_process_cast_output1(path):
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


def post_process_cast_output(path):
    with open(path, 'r') as file1:
        data = file1.readlines()
        print(len(data))
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

                print(i)
                # print(re.sub('\D', '_', i))
                region_start = int(i[18:].split('to')[0].replace(" ", ""))
                region_end = int(i.split('to')[-1].split('corrected')[0].replace(" ", ""))
                # print(i[18:],region_start)
                idpr += f"{region_start},{region_end},"
                print(f"{region_start},{region_end},")
            elif '>' in i:
                regions = []
                protein_ids.insert(protein_count, i)
                proteins.insert(protein_count, protein_seq)
                idp_regions.insert(protein_count, idpr)
                protein_count += 1
                print(i, protein_seq)
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
        proteins.append(protein_seq)
        proteins.pop(0)
        idp_regions.append(idpr)
        idp_regions.pop(0)
        print(len(proteins), len(protein_ids), len(idp_regions))
        print(protein_ids[-1], '\n', proteins[-1], '\n', idp_regions[-1])

        file1.close()
        with open(path + 'postprocessed.txt', 'w') as f:
            for i in range(len(proteins)):
                f.write(f"{protein_ids[i]}\n{proteins[i]}\n{idp_regions[i]}\n")
        f.close()
        for i in range(len(proteins)):
            s = (f"{protein_ids[i]}\n{proteins[i]}\n{idp_regions[i]}\n")
            sequence_len = len(proteins[i])
            print(idp_regions[i].split(','))
            regions = idp_regions[i].split(',')
            if len(regions) == 1:
                pred = np.zeros(sequence_len)
            else:
                pred = np.zeros(sequence_len)
                regions = regions[:-1]
                regions = [int(i) for i in regions]
                print(regions)
                regions = iter(regions)
                for x in regions:
                    start, end = x, next(regions)
                    pred[start:end] = 1
                print(sequence_len, pred)
            pred_string = ''
            pred = pred.tolist()
            for i in pred:
                pred_string += str(int(i))
            print(pred_string)

            # print(regions)

        return count


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

    annotations = []
    with open(ground_truth_path, 'r') as file1:
        gt = file1.read().splitlines()
        # print(gt)
        for i in gt:
            if not '>' in i:
                annotations.append(i)

    assert len(annotations) == len(predictions)
    avgf1 = 0
    avg_mcc = 0
    avg_cm = [0, 0, 0, 0]
    for i in range(len(predictions)):
        pred = [int(c) for c in predictions[i]]
        target = [int(c) for c in annotations[i]]
        # print(len(pred), len(target))
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
        print(TN / len(pred), FP / len(pred), FN / len(pred), TP / len(pred))
        avg_cm[0] += TP / len(pred)
        avg_cm[1] += TN / len(pred)
        avg_cm[2] += FP / len(pred)
        avg_cm[3] += FN / len(pred)
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
        #
        # # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        # print(ACC)
        # if ACC>1:
        #     TP

        # print(auc)
    print(avgf1 / len(predictions), avg_mcc / len(predictions))
    avg_cm[0] = avg_cm[0]  # * len(predictions)
    avg_cm[1] = avg_cm[1]  # * len(predictions)
    avg_cm[2] = avg_cm[2]  # * len(predictions)
    avg_cm[3] = avg_cm[3]  # * len(predictions)
    print(avg_cm)
    return count


def metric(dataset_preds, dataset_target):
    pred = np.array(dataset_preds)
    target = np.array(dataset_target)
    # print(pred)
    confusion_matrix = sklearn.metrics.confusion_matrix(target, pred)

    cm = confusion_matrix.ravel()
    # print(cm)

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
    ACC = (TP + TN) / (TP + FP + FN + TN)

    BAC = (TPR + TNR) / 2.0
    numerotr = TP * TN - FP * FN
    # denominator = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    MCC = (PPV * TPR * TNR * NPV) ** 0.5 - ((FDR * FNR * FPR * FOR) ** 0.5)
    s = f'\nTP\tTN\tFP\tFN\t\n{TP:.2f}\t{TN:.2f}\t{FP:.2f}\t{FN:.2f}\n' \
        f'F1\tMCC\tTPR\tTNR\tPPV\tNPV\tFPR\tFNR\tBAC\t \n' \
        f'{F1 :.4f}\t{MCC :.4f}\t{TPR:.4f}\t{TNR:.4f}\t{PPV:.4f}\t{NPV:.4f}\t{FPR:.4f}\t{FNR:.4f}\t{BAC:.4f}'
    return s# f'TP {TP:.2f} ,TN {TN:.2f} FP {FP:.2f},FN {FN:.2f},\n F1 {F1 :.4f}   MCC {MCC :.4f}\n TPR {TPR:.4f} TNR ' \
           #f'{TNR:.4f}  PPV {PPV:.4f}\nNPV {NPV:.4f} FPR {FPR:.4f} FNR {FNR:.4f} BAC {BAC:.4f}'



def target_metrics(idppreds, annotations):
    assert len(annotations) == len(idppreds), print(f'{len(annotations)}  != {len(idppreds)}')
    print(len(annotations))
    avgf1 = 0
    avg_mcc = 0
    avg_cm = [0, 0, 0, 0]
    dataset_preds = []
    dataset_target = []
    #print(idppreds)
    for i in range(len(idppreds)):
        # print(idppreds[i])
        pred = [int(c) for c in idppreds[i]]
        target = [int(c) for c in annotations[i]]
        #print(i, len(pred), len(target))
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
    #print(s)

    return s


def cast_metrics(predictions_path, ground_truth_path):
    with open(predictions_path, 'r') as file1:
        data = file1.readlines()
    file1.close()
    print(len(data))
    count = 0
    proteins = []
    predictions = []
    s = ''
    protein_count = 0
    for idx, i in enumerate(data):

        i = i.strip('\n')
        # print(f'{i}')
        if 'region' in i:
            count += 1
            # print(i)
        elif '>' in i:

            #print(s)
            #print(i)

            proteins.append(i)
            if s != '':
                predictions.append(s)
            s = ''
        else:
            s += i
    #print(s)
    if s != '':
        predictions.append(s)
    #print(len(proteins), len(predictions))
    idppreds = []
    for i in range(len(predictions)):
        #print(predictions[i])
        s = predictions[i]
        s = s.replace('X', '1')
        s = re.sub('\D', '0', s)
        # print(s)
        idppreds.append(s)
    print(len(idppreds))
    annotations = []
    with open(ground_truth_path, 'r') as file1:
        gt = file1.read().splitlines()
        # print(gt)
        for i in gt:
            if not '>' in i:
                annotations.append(i)

    assert len(annotations) == len(idppreds)
    target_metrics(idppreds, annotations)
    # print(len(annotations))
    # avgf1 = 0
    # avg_mcc = 0
    # avg_cm = [0,0,0,0]
    # for i in range(len(idppreds)):
    #     pred = [int(c) for c in idppreds[i]]
    #     target = [int(c) for c in annotations[i]]
    #     print(len(pred), len(target))
    #     assert len(pred) == len(target)
    #     pred = np.array(pred)
    #     target = np.array(target)
    #     auc = sklearn.metrics.accuracy_score(target, pred)
    #     precision, recall, thresholds = sklearn.metrics.precision_recall_curve(target, pred)
    #     f1 = sklearn.metrics.f1_score(target, pred, average='macro')
    #     mcc = sklearn.metrics.matthews_corrcoef(np.where(target < 1, -1, 1), np.where(pred < 1, -1, 1))
    #     # mcc = sklearn.metrics.matthews_corrcoef(target,pred)
    #     # print(np.where(target<1,target,-1),target)
    #     # print(precision, f1, mcc)
    #     avg_mcc += mcc
    #     avgf1 += f1
    #     confusion_matrix = sklearn.metrics.confusion_matrix(target, pred)
    #     #
    #     # FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    #     # FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    #     # TP = np.diag(confusion_matrix)
    #
    #     cm = confusion_matrix.ravel()
    #     TN, FP, FN, TP = cm
    #     avg_cm[0]+=  TP/len(pred)
    #     avg_cm[1] += TN/len(pred)
    #     avg_cm[2] += FP/len(pred)
    #     avg_cm[3] += FN/len(pred)
    #     print(TN, FP, FN, TP)
    #     # print(cm, cm.sum(), len(pred))
    #
    #     # Sensitivity, hit rate, recall, or true positive rate
    #     TPR = TP / (TP + FN)
    #     # Specificity or true negative rate
    #     TNR = TN / (TN + FP)
    #     # Precision or positive predictive value
    #     PPV = TP / (TP + FP)
    #     # Negative predictive value
    #     NPV = TN / (TN + FN)
    #     # Fall out or false positive rate
    #     FPR = FP / (FP + TN)
    #     # False negative rate
    #     FNR = FN / (TP + FN)
    #     # False discovery rate
    #     FDR = FP / (TP + FP)
    #
    #     # Overall accuracy
    #     ACC = (TP + TN) / (TP + FP + FN + TN)
    #     print(ACC)
    #
    #     # print(auc)
    # print(avgf1 / len(idppreds), avg_mcc / len(idppreds))
    # avg_cm[0] = avg_cm[0]#* len(predictions)
    # avg_cm[1] =avg_cm[1]#* len(predictions)
    # avg_cm[2] =avg_cm[2]#* len(predictions)
    # avg_cm[3] =avg_cm[3]#* len(predictions)
    # print(avg_cm)
    # print(f'TP,TN,FP,FN\n{avg_cm[0]:.2f},{avg_cm[1]:.2f},{avg_cm[2]:.2f},{avg_cm[3]:.2f}\n F1 {avgf1 / len(
    # predictions):.4f}  MCC {avg_mcc / len(predictions):.4f}')
    # return count


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

# # read_json('/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/data/DisProt release_2021_08.json')
# # import glob
# # files_ = sorted(glob.glob(f'/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/data/CAID_data_2018/**.txt'))
# # print(files_)
# # for i in files_:
# #
# #     read_caid_data(i)
#
# # post_process_cast_output('/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/results/cast/data_disprot-disorder.out.txt',
# #                          '/mnt/784C5F3A4C5EF1FC/PROJECTS/MScThesis/data/CAID_data_2018/annotation_files
# #                          /annot_disprot-disorder.txt')
#
#
# # cast_metrics_V2('../results/cast/CAID2018_out.txt',
# #                 '../data/CAID_data_2018/annotation_files/annot_disprot-disorder.txt')
# #
# cast_metrics ('../results/cast/CAID2018_out.txt',
#                '../data/CAID_data_2018/annotation_files/annot_disprot-disorder.txt')
# #
# #
# #
#metrics_seg('/home/iliask/PycharmProjects/MScThesis/results/seg/CAID2018_seg_out.txt', '../data/CAID_data_2018/annotation_files/annot_disprot-disorder.txt')
#
# #post_process_iupred2a_out('/home/iliask/PycharmProjects/MScThesis/results/iupred2a/Caid_disorder.txt',
# '../data/CAID_data_2018/annotation_files/annot_disprot-disorder.txt')


# read_disprot('/home/iliask/PycharmProjects/MScThesis/data/DisProt release_2021_08.fasta')
# ground_truth = '../data/CAID_data_2018/annotation_files/annot_disprot-disorder.txt'
# results_file = '/home/iliask/PycharmProjects/MScThesis/results/iupred2a/CAID2018__out.txt'
# annotations = read_disprot(ground_truth)
# iupred2a_metrics(results_file, annotations)
