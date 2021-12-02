import torch
import torch.nn as nn
from tape import ProteinBertModel


class IDP_ProteinBert(nn.Module):
    def __init__(self):
        super(IDP_ProteinBert, self).__init__()
        self.model = ProteinBertModel.from_pretrained('bert-base')

        self.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(768, 2))

    def forward(self, input_ids, input_mask=None):

        outputs = self.model(input_ids,
                             input_mask)
        sequence_output = outputs[0]
        # REMOVE CLS AND SEP TOKEN
        return self.fc(sequence_output)[:, 1:-1, :]
