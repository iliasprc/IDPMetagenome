import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from idp_methods.utils import *

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)


class IDP_fc(nn.Module):
    def __init__(self, input_channels=7, segments=20, hidden_dim=32, n_layers=2, classes=20):
        super().__init__()
        self.input_channels = input_channels * segments
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.classes = classes
        self.input_fc = nn.Sequential(nn.Linear(self.input_channels, self.hidden_dim), nn.ReLU())

        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim, self.classes))

    def forward(self, x):
        out = self.input_fc(x)

        return self.classifier(out)


# train_dataset = np.load('/results/mobidb/mxd494_train.npy',
#                         allow_pickle=True).item()
#
# val_dataset = np.load('/results/mobidb/mxd494_val.npy', allow_pickle=True).item()
#


def dataset_preparation():
    train_dataset = np.load('./results/mobidb/mxd494_train_pred2.npy',
                            allow_pickle=True).item()

    val_dataset = np.load('./results/mobidb/mxd494_val_pred2.npy', allow_pickle=True).item()

    predictors = ['prediction-disorder-iupl', 'prediction-disorder-iups',
                  'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
                  'prediction-disorder-glo', 'cast', 'seg']

    test_predictor = 'prediction-disorder-glo'  # 'prediction-disorder-iups'
    train_predictors = ['prediction-disorder-espD', 'prediction-disorder-iupl',
                        'prediction-disorder-iups', 'prediction-disorder-espN', 'prediction-disorder-espX', 'cast',
                        'seg']

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
        sample_data = sample_data[:, :step * segments]
        sample_data = sample_data.view(num_predictors, segments, step)

        sample_data, _ = sample_data.max(dim=-1)

        target = torch.from_numpy(train_dataset[sample][test_predictor]).unsqueeze(0)
        target = target[:, :step * segments]
        target = target.view(1, segments, step)

        target, _ = target.max(dim=-1)

        train_X.append(sample_data.float().view(-1,num_predictors))
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

    train_X = torch.stack(train_X, dim=0).view(-1,num_predictors)
    trainY = torch.stack(trainY, dim=0).view(-1)
    val_X = torch.stack(val_X, dim=0).view(-1,num_predictors)
    valY = torch.stack(valY, dim=0).view(-1)
    return train_X,trainY,val_X,valY



train_X,trainY,val_X,valY = dataset_preparation()
print(train_X.shape,trainY.shape)
m = IDP_fc()
EPOCHS = 50
optimizer = torch.optim.SGD(m.parameters(), lr=1e-3)
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
m = m.to(device)
loss = nn.MSELoss()


BATCH_SIZE = 20
for i in range(EPOCHS):
    train_loss = 0.0
    val_loss = 0.0
    yhat = []
    y = []
    for batchidx in range(len(train_X)):
        sample = train_X[batchidx].to(device).unsqueeze(0)
        target = trainY[batchidx].to(device)
        out = m(sample).squeeze(0)
        #print(sample.shape, target.shape,out.shape)


        target = target
        # print(target.shape, out.shape)
        loss_sclar = loss(out, target)
        loss_sclar.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss_sclar.item()
    #     output = torch.softmax(out, dim=-1)  # .squeeze()
    #     # print(output.shape)
    #     _, output = torch.max(output, 1)
    #     # print(output.shape)
    #     y += target.squeeze().detach().cpu().numpy().tolist()
    #     yhat += output.tolist()
    # metrics_ = metric(yhat, y)

    print(f'EPOCH {i} Train Loss {train_loss / (batchidx + 1):.7f}')

    train_loss = 0.0
    val_loss = 0.0
    yhat = []
    y = []
    for batchidx in range(len(val_X)):
        sample = val_X[batchidx].to(device)
        out = m(sample.unsqueeze(0)).squeeze(0)
        target = valY[batchidx].to(device)
        # print(target.shape,out.shape)
        target = target.mean(dim=0)
        loss_sclar = loss(out, target)
        loss_sclar.backward()
        optimizer.step()
        optimizer.zero_grad()
        val_loss += loss_sclar.item()
    #     output = torch.softmax(out, dim=-1)  # .squeeze()
    #     # print(output.shape)
    #     _, output = torch.max(output, 1)
    #     # print(output.shape)
    #     y += target.squeeze().detach().cpu().numpy().tolist()
    #     yhat += output.tolist()
    # metrics_ = metric(yhat, y)

    print(f'EPOCH {i} Val Loss {val_loss / (batchidx + 1):.7f}')

    # print(out.shape,sample.shape)
