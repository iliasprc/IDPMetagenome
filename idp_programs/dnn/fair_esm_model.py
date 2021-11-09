import esm
import torch
import torch.nn as nn
# Load ESM-1b model

class IDP_ESM(nn.Module):
    def __init__(self):
        super(IDP_ESM, self).__init__()
        self.feature_extractor, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
       # 3esm.pretrained.esm1_t6_43M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(768,2))#,nn.GELU()


    def forward(self,x):
       # print(len(x))
        # if len(x) > 1024:
        #     print(len(x))
        #     x1 = x[:1023]
        batch_labels, batch_strs, batch_tokens = self.batch_converter([('1',x)])

        #with torch.no_grad():
        results = self.feature_extractor(batch_tokens.cuda(), repr_layers=[6])

        token_representations = results["representations"][6]#.squeeze(0)#[1:len(x) ]
        #print(token_representations.shape)
        return self.fc(token_representations[:,1:,:])