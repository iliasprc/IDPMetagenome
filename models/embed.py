import torch.nn  as nn
from models.utils import weights_init
class LM(nn.Module):
    def __init__(self,vocab,dim=256):
        super(LM,self).__init__()
        self.embed = nn.Embedding(vocab,dim)
        self.fc = nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,vocab))
        self.apply(weights_init)
    def forward(self,x):
        em = self.embed(x)
        return self.fc(em)