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

#
