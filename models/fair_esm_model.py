import esm
import torch.nn as nn


# Load ESM-1b model

class IDP_esm1_t6_43M_UR50S(nn.Module):
    def __init__(self):
        super(IDP_esm1_t6_43M_UR50S, self).__init__()

        self.feature_extractor, alphabet = esm.pretrained.esm1_t6_43M_UR50S()

        self.batch_converter = alphabet.get_batch_converter()
        self.fc = nn.Sequential(nn.LayerNorm(768),nn.Dropout(0.5), nn.Linear(768, 2))

    def forward(self, x):
        # print(len(x))
        # if len(x) > 1024:
        #     print(len(x))
        #     x1 = x[:1023]
        batch_labels, batch_strs, batch_tokens = self.batch_converter([('1', x)])

        # with torch.no_grad():
        results = self.feature_extractor(batch_tokens.cuda(), repr_layers=[6])

        token_representations = results["representations"][6]  # .squeeze(0)#[1:len(x) ]
        # print(token_representations.shape)
        return self.fc(token_representations[:, 1:, :])


class IDP_esm1_t12_85M_UR50S(nn.Module):
    def __init__(self):
        super(IDP_esm1_t12_85M_UR50S, self).__init__()

        self.feature_extractor, alphabet = esm.pretrained.esm1_t12_85M_UR50S()

        self.batch_converter = alphabet.get_batch_converter()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, 2))

    def forward(self, x):

        batch_labels, batch_strs, batch_tokens = self.batch_converter([('1', x)])


        results = self.feature_extractor(batch_tokens.cuda(), repr_layers=[1,6,12])

        token_representations = 0.5*results["representations"][12] +  0.3*results["representations"][6] + 0.2* results["representations"][1]     # .squeeze(0)#[1:len(x) ]

        return self.fc(token_representations[:, 1:, :])
