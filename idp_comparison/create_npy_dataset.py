import numpy as np

from idp_methods.utils import seg_predictions, cast_metrics, read_fldpnn_file

ground_truth = '/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annotations/MXD494'
results_file = '/results/cast/mxd494_test.txt'

cast_preds = cast_metrics(results_file, ground_truth)

segpreds, _ = seg_predictions('/results/seg/mxd_494_test.txt')

proteins_ids, sequences, fldpnn_predictions = read_fldpnn_file(
    '/idp_methods/fldpnn_docker/mxd494.test/function_results.txt')

big_dict = {}
print(len(cast_preds))
assert len(cast_preds) == len(segpreds), 'oh no not equal lens'
assert len(fldpnn_predictions) == len(segpreds)
LEN = len(cast_preds)

for i in range(LEN):
    pred_cast = np.array([int(x) for x in cast_preds[i]])
    pred = {}
    pred_seg = np.array([int(x) for x in segpreds[i]])
    pred_fldpnn = np.array([int(x) for x in fldpnn_predictions[i]])
    big_dict[str(i)] = []
    pred['cast'] = pred_cast
    pred['seg'] = pred_seg
    pred['fldpnn'] = pred_fldpnn
    big_dict[str(i)] = pred

#
val_dataset = np.load('/results/mobidb/mxd494_val_pred.npy',
                      allow_pickle=True).item()
print(val_dataset.keys())

idx = 0
for k, v in val_dataset.items():

    val_dataset[str(idx)]['cast'] = big_dict[str(idx)]['cast']
    val_dataset[str(idx)]['seg'] = big_dict[str(idx)]['seg']
    val_dataset[str(idx)]['fldpnn'] = big_dict[str(idx)]['fldpnn']
    idx += 1

print(val_dataset['0'])
np.save('/home/iliask/PycharmProjects/MScThesis/results/mobidb/mxd494_val_pred3.npy ', val_dataset)
