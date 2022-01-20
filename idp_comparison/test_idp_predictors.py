import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from models.rnn import IDP_test_rnn

from idp_methods.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--dataset', type=str, default="d723", help='dataset name')
parser.add_argument('--epochs', type=int, default=50, help='total number of epochs')
parser.add_argument('--test-predictor', type=str, default='seg',
                    choices=['prediction-disorder-iupl', 'prediction-disorder-iups',
                             'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
                             'prediction-disorder-glo', 'cast', 'seg'])

args = parser.parse_args()

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
dataset = args.dataset

if dataset == 'd723':
    train_dataset = np.load('./results/mobidb/d723_train2.npy',
                            allow_pickle=True).item()

    val_dataset = np.load('./results/mobidb/d723_test2.npy', allow_pickle=True).item()
    print(val_dataset['0'].keys())
    predictors = ['prediction-disorder-iupl', 'prediction-disorder-iups',
                  'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
                  'prediction-disorder-glo', 'cast', 'seg']
elif dataset == 'mxd494':


    train_dataset = np.load('./results/mobidb/mxd494_train_pred3.npy',
                            allow_pickle=True).item()

    val_dataset = np.load('./results/mobidb/mxd494_val_pred3.npy', allow_pickle=True).item()
    print(val_dataset['0'].keys())
    predictors = ['prediction-disorder-iupl', 'prediction-disorder-iups',
                  'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
                  'prediction-disorder-glo', 'cast', 'seg','fldpnn']

test_predictor = args.test_predictor
predictors.remove(test_predictor)
train_predictors = predictors
assert len(train_predictors) == len(predictors)

def next_number(x, N=20):
    if x % 20:
        return x + (N - x % N)
    else:
        return 0



m = IDP_test_rnn(input_channels=len(train_predictors))
train_X = []
trainY = []
for sample in train_dataset:
    # print(train_dataset[sample].keys())
    sample_data = torch.tensor([])
    for preds_ in train_predictors:
        data = torch.from_numpy(train_dataset[sample][preds_]).unsqueeze(0).float()
        sample_data = torch.cat([sample_data, data])

    train_X.append(sample_data.transpose(0, 1).float())
    trainY.append(torch.from_numpy(train_dataset[sample][test_predictor]).unsqueeze(0).transpose(0, 1).float())
    # print(torch.from_numpy(train_dataset[sample][test_predictor]).unsqueeze(0).shape,sample_data.shape)
val_X = []
valY = []
for sample in val_dataset:
    # print(train_dataset[sample].keys())
    sample_data = torch.tensor([])
    for preds_ in train_predictors:
        data = torch.from_numpy(train_dataset[sample][preds_]).unsqueeze(0).float()
        sample_data = torch.cat([sample_data, data])

    val_X.append(sample_data.transpose(0, 1).float())
    valY.append(torch.from_numpy(train_dataset[sample][test_predictor]).unsqueeze(0).transpose(0, 1).float())

EPOCHS = 50
optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
m = m.to(device)
loss = nn.CrossEntropyLoss()
for i in range(EPOCHS):
    train_loss = 0.0
    val_loss = 0.0
    yhat = []
    y = []
    for batchidx in range(len(train_X)):
        sample = train_X[batchidx].to(device)
        out = m(sample.unsqueeze(0)).squeeze(0)
        target = trainY[batchidx].squeeze(-1).to(device)
        # print(target.shape,out.shape)
        loss_sclar = loss(out, target.long())
        loss_sclar.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss_sclar.item()
        output = torch.softmax(out, dim=-1)  # .squeeze()
        # print(output.shape)
        _, output = torch.max(output, 1)
        # print(output.shape)
        y += target.squeeze().detach().cpu().numpy().tolist()
        yhat += output.tolist()
        #print(f'EPOCH {i} Train Loss {train_loss / (batchidx + 1):.4f}')
    metrics_, _ = dataset_metrics(yhat, y)

    print(f'EPOCH {i} Train Loss {train_loss / (batchidx + 1):.4f}')
    print(f'EPOCH TRAIN METRICS{i}\n{metrics_}')
    train_loss = 0.0
    val_loss = 0.0
    yhat = []
    y = []
    for batchidx in range(len(val_X)):
        sample = val_X[batchidx].to(device)
        out = m(sample.unsqueeze(0)).squeeze(0)
        target = valY[batchidx].squeeze(-1).to(device)
        # print(target.shape,out.shape)
        loss_sclar = loss(out, target.long())
        loss_sclar.backward()
        optimizer.step()
        optimizer.zero_grad()
        val_loss += loss_sclar.item()
        output = torch.softmax(out, dim=-1)  # .squeeze()
        # print(output.shape)
        _, output = torch.max(output, 1)
        # print(output.shape)
        y += target.squeeze().detach().cpu().numpy().tolist()
        yhat += output.tolist()
    metrics_, _ = dataset_metrics(yhat, y)

    print(f'EPOCH {i} Val Loss {val_loss / (batchidx + 1):.4f}')
    print(f'EPOCH VALIDATION METRICS {i}\n{metrics_}')
    # print(out.shape,sample.shape)
