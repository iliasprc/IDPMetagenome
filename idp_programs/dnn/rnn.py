from torch import nn

from .tcn import TemporalConvNet
from einops import rearrange
class IDPrnn(nn.Module):
    def __init__(self, dim, blocks=2, dropout=0.5, classes=2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(20, dim),nn.LeakyReLU(0.1))
        self.tcn = TemporalConvNet(num_inputs=dim, num_channels=[dim,dim//2,dim//2,dim], kernel_size=2, dropout=0.2)

        self.rnn = nn.LSTM(batch_first=True, input_size=dim, hidden_size=dim, num_layers=blocks, dropout=dropout,
                           bidirectional=True)

        self.head = nn.Linear(2 * dim, classes)

    def forward(self, x, mask=None):
        # print(x.shape)
        # assert len(x.shape) == 3
        x = self.embed(x)
        # print(x.shape)
        x = rearrange(x,'b t c -> b c t')
        x = self.tcn(x)
        x = rearrange(x, ' b c t -> b t c')
        x,_ = self.rnn(x)
        #print(x.shape)
        x = self.head(x)#.squeeze(-1)
        return x

#
