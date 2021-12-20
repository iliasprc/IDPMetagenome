from torch import nn


class IDPrnn(nn.Module):
    def __init__(self, dim, blocks=2, dropout=0.5, classes=2,embed_dim=30):
        super().__init__()
        self.embed = nn.Embedding(embed_dim, dim)

        self.rnn = nn.LSTM(batch_first=True, input_size=dim, hidden_size=dim, num_layers=blocks, dropout=dropout,
                           bidirectional=True)

        self.head = nn.Linear(2 * dim, classes)

    def forward(self, x, mask=None):

        x = self.embed(x)

        x, _ = self.rnn(x)

        x = self.head(x)
        return x



class IDP_test_rnn(nn.Module):
    def __init__(self, input_channels=7, hidden_dim=32, n_layers=2, classes=2, segments=20):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.classes = classes
        self.input_fc = nn.Linear(self.input_channels, self.hidden_dim)
        self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers,
                           bidirectional=True, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(2 * self.hidden_dim, self.classes)

    def forward(self, x):
        # print(x.shape)
        x = self.input_fc(x)
        out, (h, c) = self.rnn(x)
        return self.classifier(out)
#
