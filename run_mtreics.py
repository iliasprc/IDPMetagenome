import numpy as np

from idp_methods.utils import seg_predictions, cast_metrics

ground_truth = '/home/iliask/PycharmProjects/MScThesis/data/idp_seq_2_seq/mxd494/annotations/MXD494'
results_file = '/home/iliask/PycharmProjects/MScThesis/results/cast/mxd494_train.txt'

cast_preds = cast_metrics(results_file, ground_truth)

segpreds, _ = seg_predictions('/home/iliask/PycharmProjects/MScThesis/results/seg/mxd494_train.txt')
big_dict = {}
print(len(cast_preds))
assert len(cast_preds) == len(segpreds), 'oh no not equal lens'
LEN = len(cast_preds)

for i in range(LEN):
    pred_cast = np.array([int(x) for x in cast_preds[i]])
    pred = {}
    pred_seg = np.array([int(x) for x in segpreds[i]])
    big_dict[str(i)]=[]
    pred['cast'] = pred_cast
    pred['seg'] = pred_seg
    big_dict[str(i)]= pred


#
val_dataset = np.load('/home/iliask/PycharmProjects/MScThesis/results/mobidb/mxd494_train.npy',
                      allow_pickle=True).item()
print(val_dataset.keys())
print(val_dataset['0'])

idx = 0
for k, v in val_dataset.items():

    val_dataset[str(idx)]['cast'] = big_dict[str(idx)]['cast']
    val_dataset[str(idx)]['seg'] = big_dict[str(idx)]['seg']
    idx += 1
#
np.save('/home/iliask/PycharmProjects/MScThesis/results/mobidb/mxd494_train_pred2.npy', val_dataset)
