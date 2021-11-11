from torch import nn


class IDPrnn(nn.Module):
    def __init__(self, dim, blocks=2, dropout=0.5, classes=2):
        super().__init__()
        self.embed = nn.Embedding(25, dim)

        self.rnn = nn.LSTM(batch_first=True, input_size=dim, hidden_size=dim, num_layers=blocks, dropout=dropout,
                           bidirectional=True)

        self.head = nn.Linear(2 * dim, classes)

    def forward(self, x, mask=None):
        # print(x.shape)
        # assert len(x.shape) == 3
        x = self.embed(x)
        # print(x.shape)

        x, _ = self.rnn(x)
        # print(x.shape)
        x = self.head(x)  # .squeeze(-1)
        return x

#
