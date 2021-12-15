import torch
import torch.nn as nn

from idp_methods.utils import *

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

train_dataset = np.load('/results/mobidb/mxd494_train_pred2.npy',
                        allow_pickle=True).item()

val_dataset = np.load('/results/mobidb/mxd494_val_pred2.npy', allow_pickle=True).item()
print(val_dataset['0'].keys())
predictors = ['prediction-disorder-iupl', 'prediction-disorder-iups',
              'prediction-disorder-espN', 'prediction-disorder-espX', 'prediction-disorder-espD',
              'prediction-disorder-glo', 'cast', 'seg']

test_predictor = 'prediction-disorder-glo'  # 'prediction-disorder-iups'
train_predictors = ['prediction-disorder-espD', 'prediction-disorder-iupl',
                    'prediction-disorder-iups', 'prediction-disorder-espN', 'prediction-disorder-espX', 'cast', 'seg']

exit()


class IDP_tester(nn.Module):
    def __init__(self, input_channels=5, hidden_dim=32, n_layers=2, classes=2):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.classes = classes
        self.input_fc = nn.Linear(self.input_channels, self.hidden_dim)
        self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers,
                           bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(2 * self.hidden_dim, self.classes)

    def forward(self, x):
        x = self.input_fc(x)
        out, (h, c) = self.rnn(x)
        return self.classifier(out)


class IDP_fc(nn.Module):
    def __init__(self, input_channels=5, hidden_dim=32, n_layers=2, classes=2):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.classes = classes
        self.input_fc = nn.Linear(self.input_channels, self.hidden_dim)

        self.classifier = nn.Linear(self.hidden_dim, self.classes)

    def forward(self, x):
        x_mean = x.mean(dim=1)
        out = self.input_fc(x_mean)

        return self.classifier(out)


m = IDP_tester()
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

EPOCHS = 10
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
    metrics_, _ = dataset_metrics(yhat, y)

    print(f'EPOCH {i} Train Loss {train_loss / (batchidx + 1):.4f}')
    print(f'EPOCH {i}\n{metrics_}')
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
    print(f'EPOCH {i}\n{metrics_}')
    # print(out.shape,sample.shape)
