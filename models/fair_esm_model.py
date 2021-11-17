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
        self.fc = nn.Sequential(nn.LayerNorm(768),nn.Dropout(0.5), nn.Linear(768, 2))

    def forward(self, x):

        batch_labels, batch_strs, batch_tokens = self.batch_converter([('1', x)])


        results = self.feature_extractor(batch_tokens.cuda(), repr_layers=[1,6,12])

        token_representations = results["representations"][12] #+  0.3*results["representations"][6] + 0.2* results["representations"][1]     # .squeeze(0)#[1:len(x) ]
        #token_representations = results["representations"][1]
        return self.fc(token_representations[:, 1:, :])



class IDP_esm1_msa(nn.Module):
    def __init__(self):
        super(IDP_esm1_msa, self).__init__()

        self.feature_extractor, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()

        self.batch_converter = alphabet.get_batch_converter()
        self.fc = nn.Sequential(nn.LayerNorm(768),nn.Dropout(0.5), nn.Linear(768, 2))

    def forward(self, x):
        #print(x)
        n = 1024 # chunk length
        chunks = [(str(i),x[i:i + n]) for i in range(0, len(x), n)]
        #print(chunks)

        batch_labels, batch_strs, batch_tokens = self.batch_converter(chunks)

        if len(x)>1024:
            print(chunks,batch_tokens)
        results = self.feature_extractor(batch_tokens.cuda(), repr_layers=[12])

        token_representations = results["representations"][12].squeeze(0) #+  0.3*results["representations"][6] + 0.2* results["representations"][1]     # .squeeze(0)#[1:len(x) ]
        #token_representations = results["representations"][1]
        #print(token_representations.shape)
        return self.fc(token_representations[:, 1:, :])
