import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from idp_methods.utils import *
from models.rnn import IDP_test_rnn

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--dataset', type=str, default="MXD494", help='dataset name')
parser.add_argument('--epochs', type=int, default=50, help='total number of epochs')
parser.add_argument('--test-predictor', type=str, default='prediction-disorder-glo',
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

train_dataset = np.load('./results/mobidb/mxd494_train_pred2.npy',
                        allow_pickle=True).item()

val_dataset = np.load('./results/mobidb/mxd494_val_pred2.npy', allow_pickle=True).item()
print(val_dataset['0'].keys())
predictors = ['prediction-disorder-iupl', 'prediction-disorder-iups',
              'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
              'prediction-disorder-glo', 'cast', 'seg']

test_predictor = args.test_predictor
predictors.remove(test_predictor)
train_predictors = predictors
assert len(train_predictors) == len(predictors)
num_predictors = len(train_predictors)


def next_number(x, N=20):
    if x % 20:
        return x + (N - x % N)
    else:
        return 0




m = IDP_test_rnn(input_channels=len(train_predictors))


def dataset_preparation_padded():
    train_dataset = np.load('./results/mobidb/mxd494_train_pred2.npy',
                            allow_pickle=True).item()

    val_dataset = np.load('./results/mobidb/mxd494_val_pred2.npy', allow_pickle=True).item()

    predictors = ['prediction-disorder-iupl', 'prediction-disorder-iups',
                  'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
                  'prediction-disorder-glo', 'cast', 'seg']

    num_predictors = len(train_predictors)

    segments = 20

    train_X = []
    trainY = []

    for sample in train_dataset:

        sample_data = torch.tensor([])
        for preds_ in train_predictors:
            data = torch.from_numpy(train_dataset[sample][preds_]).unsqueeze(0).float()

            sample_data = torch.cat([sample_data, data])

        length = sample_data.shape[-1]
        step = int(length / segments)
        # print(sample_data.shape)
        length = sample_data.shape[-1]

        pad_ = torch.zeros((sample_data.shape[0], next_number(length) - length))

        sample_data = torch.cat((sample_data, pad_), dim=-1)

        sample_data = sample_data[:, :step * segments]
        sample_data = sample_data.view(num_predictors, segments, step)

        sample_data, _ = sample_data.max(dim=-1)

        target = torch.from_numpy(train_dataset[sample][test_predictor]).unsqueeze(0)

        pad_ = torch.zeros((target.shape[0], next_number(length) - length))
        target = torch.cat((target, pad_), dim=-1)
        print(target.shape)
        target = target[:, :step * segments]
        # print(target.tolist())

        target = target.view(1, segments, step)

        target, _ = target.max(dim=-1)
        # print(target.tolist())

        train_X.append(sample_data.float().view(-1, num_predictors))
        trainY.append(target.float().view(-1))

    val_X = []
    valY = []
    for sample in val_dataset:
        # print(train_dataset[sample].keys())
        sample_data = torch.tensor([])
        for preds_ in train_predictors:
            data = torch.from_numpy(val_dataset[sample][preds_]).unsqueeze(0).float()
            sample_data = torch.cat([sample_data, data])

        length = sample_data.shape[-1]
        pad_ = torch.zeros((sample_data.shape[0], next_number(length) - length))
        sample_data = torch.cat((sample_data, pad_), dim=-1)
        step = int(length / segments)
        sample_data = sample_data[:, :step * segments]
        sample_data = sample_data.view(num_predictors, segments, step)
        sample_data, _ = sample_data.max(dim=-1)

        target = torch.from_numpy(val_dataset[sample][test_predictor]).unsqueeze(0)
        pad_ = torch.zeros((target.shape[0], next_number(length) - length))
        target = torch.cat((target, pad_), dim=-1)
        target = target[:, :step * segments]
        target = target.view(1, segments, step)

        target, _ = target.max(dim=-1)

        val_X.append(sample_data.float())
        valY.append(target.float().view(-1))

    # train_X = torch.stack(train_X, dim=0).view(-1,num_predictors)
    # trainY = torch.stack(trainY, dim=0).view(-1)
    # val_X = torch.stack(val_X, dim=0).view(-1,num_predictors)
    # valY = torch.stack(valY, dim=0).view(-1)

    train_X = torch.stack(train_X, dim=0)  # .view(-1, num_predictors)
    trainY = torch.stack(trainY, dim=0)  # .view(-1)
    val_X = torch.stack(val_X, dim=0).permute(0, 2, 1)  # .view(-1, num_predictors)
    valY = torch.stack(valY, dim=0)  # .view(-1)
    return train_X, trainY, val_X, valY


def dataset_preparation():
    train_dataset = np.load('./results/mobidb/mxd494_train_pred2.npy',
                            allow_pickle=True).item()

    val_dataset = np.load('./results/mobidb/mxd494_val_pred2.npy', allow_pickle=True).item()

    segments = 20

    train_X = []
    trainY = []

    for sample in train_dataset:

        sample_data = torch.tensor([])
        for preds_ in train_predictors:
            data = torch.from_numpy(train_dataset[sample][preds_]).unsqueeze(0).float()

            sample_data = torch.cat([sample_data, data])

        length = sample_data.shape[-1]
        step = int(length / segments)
        print(sample_data.shape)
        sample_data = sample_data[:, :step * segments]
        sample_data = sample_data.view(num_predictors, segments, step)

        sample_data, _ = sample_data.max(dim=-1)

        target = torch.from_numpy(train_dataset[sample][test_predictor]).unsqueeze(0)
        target = target[:, :step * segments]
        target = target.view(1, segments, step)

        target, _ = target.max(dim=-1)

        train_X.append(sample_data.float().view(-1, num_predictors))
        trainY.append(target.float().view(-1))

    val_X = []
    valY = []
    for sample in val_dataset:
        # print(train_dataset[sample].keys())
        sample_data = torch.tensor([])
        for preds_ in train_predictors:
            data = torch.from_numpy(val_dataset[sample][preds_]).unsqueeze(0).float()
            sample_data = torch.cat([sample_data, data])

        length = sample_data.shape[-1]
        step = int(length / segments)
        sample_data = sample_data[:, :step * segments]
        sample_data = sample_data.view(num_predictors, segments, step)
        sample_data, _ = sample_data.max(dim=-1)

        target = torch.from_numpy(val_dataset[sample][test_predictor]).unsqueeze(0)
        target = target[:, :step * segments]
        target = target.view(1, segments, step)

        target, _ = target.max(dim=-1)

        val_X.append(sample_data.float())
        valY.append(target.float().view(-1))

    train_X = torch.stack(train_X, dim=0)  # .view(-1, num_predictors)
    trainY = torch.stack(trainY, dim=0)  # .view(-1)
    val_X = torch.stack(val_X, dim=0).permute(0, 2, 1)  # .view(-1, num_predictors)
    valY = torch.stack(valY, dim=0)  # .view(-1)
    return train_X, trainY, val_X, valY


train_X, trainY, val_X, valY = dataset_preparation()

print(f'training set X {train_X.shape} Y {trainY.shape}')
print(f'validation set X {val_X.shape} Y {valY.shape}')

EPOCHS = 50
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
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
