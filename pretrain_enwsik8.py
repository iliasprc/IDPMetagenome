# constants
import argparse
import datetime
import gzip
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# d = torchtext.datasets.EnWik9(root='.dataenwik9', split=('train', ))
#
# exit()
parser = argparse.ArgumentParser(description='PyTorch  Language Model')
parser.add_argument('--dataset', type=str, default='enwik8')
parser.add_argument('--data', type=str,
                    default='/home/iliask/PycharmProjects/MScThesis/data/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='Reformer',
                    help='type of  net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer,Reformer)')
parser.add_argument('--n_hashes', type=int, default=4)
parser.add_argument('--nhead', type=int, default=46,

                    help='the number of heads in the encoder/decoder of the transformer model')

parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--depth', type=int, default=6, help='number of layers')
parser.add_argument('--gradient_steps', type=int, default=32)
parser.add_argument('--causal', action='store_true', default=False)
parser.add_argument('--tied_connections', action='store_true', default=False)
parser.add_argument('--kmeans', action='store_true', default=False)
parser.add_argument('--full_attention', action='store_true', default=False)
parser.add_argument('--seqlen', type=int, default=1024,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', default=True,
                    help='tie the word embedding and softmax weights')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')

parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--cpkt_dir', type=str, default='./cpktsenwik8',
                    help='checkpoint directory')
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATE_EVERY = args.gradient_steps
LEARNING_RATE = args.lr
VALIDATE_EVERY = 10000
GENERATE_EVERY = 2500
SEQ_LEN = args.seqlen
GENERATE_LENGTH = SEQ_LEN
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")


# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


# instantiate model
def select_model(args, name, n_classes, pretrained=False):
    dim = args.emsize
    if name == 'idptransformer':
        from models.transformer import IDPTransformer
        return IDPTransformer(dim=dim, blocks=args.depth, heads=args.nhead, dim_head=None, dim_linear_block=dim * 2,
                              dropout=0.1,
                              prenorm=False, classes=n_classes)
    elif name == 'idpcct':
        from models.transformer import IDP_cct
        return IDP_cct(dim=dim, blocks=args.depth, heads=args.nhead, dim_head=None, dim_linear_block=dim * 2,
                       dropout=0.2,
                       prenorm=False, classes=n_classes)

name = 'idpcct'
model = select_model(args, 'idpcct', 256)
if use_cuda:
    model.cuda()

time_string = datetime.datetime.now().strftime("%d_%m_%Y_%H.%M.%S")

pathdir = os.path.join(args.cpkt_dir, time_string,name)
# prepare enwik8 data
writer = SummaryWriter(pathdir + '/runs')

with gzip.open(args.data + 'enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq  # .cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len


train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

len_epoch = len(train_loader) * BATCH_SIZE

print(len(train_loader))
# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
from models.utils import Cosine_LR_Scheduler
scheduler = Cosine_LR_Scheduler(
        optim,
        warmup_epochs=3, warmup_lr=0,
        num_epochs=EPOCHS, base_lr=LEARNING_RATE, final_lr=1e-5,
        iter_per_epoch=len(train_loader)//GRADIENT_ACCUMULATE_EVERY,
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

print(model)
# training
best_loss = 1000
idx = 0

for i in range(EPOCHS):

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    trainloss = 0
    for idx, data in enumerate(train_loader):
        # data=data.unsqueeze(-1)
        target = data[:, 1:].to(device)
        data = data[:, 0:-1].to(device)
        # print(data.shape)
        output = model(data)
        b, t, _ = output.shape
        output = output.view(b * t, -1)
        target = target.reshape(-1)
        # print(output.shape,target.shape)
        loss = criterion(output, target)
        writer_step = (i - 1) * len_epoch + idx
        writer.add_scalar('Train/Loss', loss.item(), writer_step)
        # print(f'Train loss {trainloss / (idx + 1)} batch {idx}/ {len(train_loader)}')
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()
        trainloss += loss.item()
        # if idx % VALIDATE_EVERY
        if idx % GRADIENT_ACCUMULATE_EVERY == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scheduler.step()
            optim.step()
            optim.zero_grad()
        if idx % 1000 == 0:
            print(f'Train loss {trainloss / (idx + 1)} batch {idx}/ {len(train_loader)}')
        if idx % VALIDATE_EVERY == 0:
            print(f'Train loss {trainloss / (idx + 1)} batch {idx}/ {len(train_loader)}')
            model.eval()
            valloss = 0
            with torch.no_grad():
                for validx, data in enumerate(val_loader):

                    target = data[:, 1:].to(device)
                    data = data[:, 0:-1].to(device)
                    # print(data.shape)
                    output = model(data)
                    b, t, _ = output.shape
                    output = output.view(b * t, -1)
                    target = target.reshape(-1)
                    # print(output.shape, target.shape)
                    loss = criterion(output, target)
                    writer.add_scalar('Val/Loss', loss.item(), writer_step)
                    valloss += loss.item()
                print(f'VAL LOSS {valloss / validx} ')
                if valloss < best_loss:
                    print('BEST'
                          )
                    best_loss = valloss
                    torch.save(model.state_dict(),
                               pathdir + f'/bestmodel.pth')
                    with open(pathdir + '/commandline_args.txt', 'w') as f:
                        json.dump(args.__dict__, f, indent=2)
                best_loss = valloss
                torch.save(model.state_dict(),
                           pathdir + f'/lastmodel.pth')
            model.train()
