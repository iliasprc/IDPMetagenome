import math
import os
import textwrap

PATH = os.path.dirname(os.path.realpath(__file__))


def avg(lst):
    return sum(lst) / len(lst)


def aa_freq(_seq):
    _freq = {}
    for _aa in _seq:
        if _aa in _freq:
            _freq[_aa] += 1
        else:
            _freq[_aa] = 1
    for _aa, _ins in _freq.items():
        _freq[_aa] = _ins / len(_seq)
    return _freq


def read_matrix(matrix_file):
    _mtx = {}
    with open(matrix_file, "r") as _fhm:
        for _line in _fhm:
            if _line.split()[0] in _mtx:
                _mtx[_line.split()[0]][_line.split()[1]] = float(_line.split()[2])
            else:
                _mtx[_line.split()[0]] = {}
                _mtx[_line.split()[0]][_line.split()[1]] = float(_line.split()[2])
    return _mtx


def read_histo(histo_file):
    hist = []
    h_min = float("inf")
    h_max = -float("inf")
    with open(histo_file, "r") as fnh:
        for _line in fnh:
            if _line.startswith("#"):
                continue
            if float(_line.split()[1]) < h_min:
                h_min = float(_line.split()[1])
            if float(_line.split()[1]) > h_max:
                h_max = float(_line.split()[1])
            hist.append(float(_line.split()[-1]))
    h_step = (h_max - h_min) / (len(hist))
    return hist, h_min, h_max, h_step


def smooth(energy_list, window):
    weighted_energy_score = [0] * len(energy_list)
    for idx in range(len(energy_list)):
        weighted_energy_score[idx] = avg(energy_list[max(0, idx - window):min(len(energy_list), idx + window + 1)])
    return weighted_energy_score


def read_seq(fasta_file):
    _seq = ""
    with open(fasta_file) as file_handler:
        for _line in file_handler:
            if _line.startswith(">"):
                continue
            _seq += _line.strip()
    return _seq


def iupred(seq, mode='long', new_smoothing=False):
    if mode == "short":
        lc = 1
        uc = 25
        wc = 10
        mtx = read_matrix("{}/data/iupred2_short_energy_matrix".format(PATH))
        histo, histo_min, histo_max, histo_step = read_histo("{}/data/short_histogram".format(PATH))

    elif mode == 'glob':
        lc = 1
        uc = 100
        wc = 15
        mtx = read_matrix("{}/data/iupred2_long_energy_matrix".format(PATH))
        histo, histo_min, histo_max, histo_step = read_histo("{}/data/long_histogram".format(PATH))

    else:
        lc = 1
        uc = 100
        wc = 10
        mtx = read_matrix("{}/data/iupred2_long_energy_matrix".format(PATH))
        histo, histo_min, histo_max, histo_step = read_histo("{}/data/long_histogram".format(PATH))

    unweighted_energy_score = [0] * len(seq)
    weighted_energy_score = [0] * len(seq)
    iupred_score = [0] * len(seq)

    for idx in range(len(seq)):
        freq_dct = aa_freq(seq[max(0, idx - uc):max(0, idx - lc)] + seq[idx + lc + 1:idx + uc + 1])
        for aa, freq in freq_dct.items():
            try:
                unweighted_energy_score[idx] += mtx[seq[idx]][aa] * freq
            except KeyError:
                unweighted_energy_score[idx] += 0

    if mode == 'short':
        for idx in range(len(seq)):
            for idx2 in range(idx - wc, idx + wc + 1):
                if idx2 < 0 or idx2 >= len(seq):
                    weighted_energy_score[idx] += -1.26
                else:
                    weighted_energy_score[idx] += unweighted_energy_score[idx2]
            weighted_energy_score[idx] /= len(range(idx - wc, idx + wc + 1))
    else:
        weighted_energy_score = smooth(unweighted_energy_score, wc)
        if new_smoothing:
            weighted_energy_score = smooth(weighted_energy_score, 15)

    glob_text = ""
    if mode == 'glob':
        gr = []
        in_gr = False
        beg, end = 0, 0
        for idx, val in enumerate(weighted_energy_score):
            if in_gr and val <= 0.3:
                gr.append({0: beg, 1: end})
                in_gr = False
            elif in_gr:
                end += 1
            if val > 0.3 and not in_gr:
                beg = idx
                end = idx
                in_gr = True
        if in_gr:
            gr.append({0: beg, 1: end})
        mgr = []
        k = 0
        kk = k + 1
        if gr:
            beg = gr[0][0]
            end = gr[0][1]
        nr = len(gr)
        while k < nr:
            if kk < nr and gr[kk][0] - end < 45:
                beg = gr[k][0]
                end = gr[kk][1]
                kk += 1
            elif end - beg + 1 < 35:
                k += 1
                if k < nr:
                    beg = gr[k][0]
                    end = gr[k][1]
            else:
                mgr.append({0: beg, 1: end})
                k = kk
                kk += 1
                if k < nr:
                    beg = gr[k][0]
                    end = gr[k][1]
        seq = seq.lower()
        nr = 0
        res = ""
        for i in mgr:
            res += seq[nr:i[0]] + seq[i[0]:i[1] + 1].upper()
            nr = i[1] + 1
        res += seq[nr:]
        res = " ".join([res[i:i + 10] for i in range(0, len(res), 10)])
        glob_text += "Number of globular domains: {}\n".format(len(mgr))
        for n, i in enumerate(mgr):
            glob_text += "          globular domain   {}.\t{}-{}\n".format(n + 1, i[0] + 1, i[1] + 1)
        glob_text += "\n".join(textwrap.wrap(res, 70))

    for idx, val in enumerate(weighted_energy_score):
        if val <= histo_min + 2 * histo_step:
            iupred_score[idx] = 1
        elif val >= histo_max - 2 * histo_step:
            iupred_score[idx] = 0
        else:
            iupred_score[idx] = histo[int((weighted_energy_score[idx] - histo_min) * (1 / histo_step))]
    return iupred_score, glob_text


def iupred_redox(seq):
    return iupred(seq.replace("C", "S"))


def get_redox_regions(redox_values, iupred_values):
    """
    Calculate the redox sensitive regions
    :param redox_values: Redox Y coordinates
    :param iupred_values: IUPred Y coordiantes
    :return:
    """
    patch_loc = {}
    trigger = False
    opening_pos = []
    start, end = 0, 0
    counter = 0
    # Calculate possible position
    for idx, redox_val in enumerate(redox_values):
        if redox_val > 0.5 > iupred_values[idx] and redox_val - iupred_values[idx] > 0.3:
            opening_pos.append(idx)
    # Filter out where not enough possible position is found
    # Enlarge region where enough position is found
    for idx, redox_val in enumerate(redox_values):
        if redox_val - iupred_values[idx] > 0.15 and redox_val >= 0.35:
            if not trigger:
                start = idx
                trigger = True
            if idx in opening_pos:
                counter += 1
            end = idx
        else:
            trigger = False
            if end - start > 14 and counter > 2:
                patch_loc[start] = end
            counter = 0
    if end - start > 14 and counter > 2:
        patch_loc[start] = end
    # Combine close regions
    deletable = []
    for start, end in patch_loc.items():
        for start2, end2 in patch_loc.items():
            if start != start2 and start2 - end < 10 and start2 > start:
                patch_loc[start] = end2
                deletable.append(start2)
    for start in deletable:
        del patch_loc[start]
    return patch_loc


def anchor2(seq):
    local_window_size = 41
    iupred_window_size = 30
    local_smoothing_window = 5
    par_a = 0.0013
    par_b = 0.26
    par_c = 0.43
    iupred_limit = par_c - (par_a / par_b)
    mtx = read_matrix('{}/data/anchor2_energy_matrix'.format(PATH))
    interface_comp = {}
    with open('{}/data/anchor2_interface_comp'.format(PATH)) as _fn:
        for line in _fn:
            interface_comp[line.split()[1]] = float(line.split()[2])
    iupred_scores = iupred(seq, new_smoothing=False)[0]
    local_energy_score = [0] * len(seq)
    interface_energy_score = [0] * len(seq)
    energy_gain = [0] * len(seq)
    for idx in range(len(seq)):
        freq_dct = aa_freq(
            seq[max(0, idx - local_window_size):max(0, idx - 1)] + seq[idx + 2:idx + local_window_size + 1])
        for aa, freq in freq_dct.items():
            try:
                local_energy_score[idx] += mtx[seq[idx]][aa] * freq
            except KeyError:
                local_energy_score[idx] += 0
        for aa, freq in interface_comp.items():
            try:
                interface_energy_score[idx] += mtx[seq[idx]][aa] * freq
            except KeyError:
                interface_energy_score[idx] += 0
        energy_gain[idx] = local_energy_score[idx] - interface_energy_score[idx]
    iupred_scores = smooth(iupred_scores, iupred_window_size)
    energy_gain = smooth(smooth(energy_gain, local_smoothing_window), local_smoothing_window)
    anchor_score = [0] * len(seq)
    for idx in range(len(seq)):
        sign = 1
        if energy_gain[idx] < par_b and iupred_scores[idx] < par_c:
            sign = -1
        corr = 0
        if iupred_scores[idx] > iupred_limit and energy_gain[idx] < 0:
            corr = (par_a / (iupred_scores[idx] - par_c)) + par_b
        anchor_score[idx] = sign * (energy_gain[idx] + corr - par_b) * (iupred_scores[idx] - par_c)
        anchor_score[idx] = 1 / (1 + math.e ** (-22.97968 * (anchor_score[idx] - 0.0116)))
    return anchor_score


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
    annotations = read_annotation_file(ground_truth_path)
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


def iupred2a_predictions(path):
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
                # print(i.split('\t'))
                amino_score = i.split('\t')[-1]
                pred += amino_score


            else:

                idppreds.append(pred)
                pred = ''
                # print(i)
    # annotations = []
    idppreds.pop(0)
    idppreds.append(pred)
    return idppreds
