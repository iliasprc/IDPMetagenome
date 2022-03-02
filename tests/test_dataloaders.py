import torch
import os
from torch.utils.data import DataLoader, RandomSampler, Dataset
from dataloaders.utils._sampler import BucketBatchSampler
from dataloaders.idp_dataloader import DM_idp_dataset
from dataloaders.tokenizer import TAPETokenizer
def setup_loader(dataset: Dataset,
                 batch_size: int,
                 local_rank: int,
                 n_gpu: int,
                 gradient_accumulation_steps: int,
                 num_workers: int) -> DataLoader:
    sampler =   RandomSampler(dataset)
    # batch_size = get_effective_batch_size(
    #     batch_size, local_rank, n_gpu, gradient_accumulation_steps) * n_gpu
    # WARNING: this will fail if the primary sequence is not the first thing the dataset returns
    batch_sampler = BucketBatchSampler(
        sampler, batch_size, False, lambda x: len(x[0]), dataset)

    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,  # type: ignore
        batch_sampler=batch_sampler)

    return loader

cwd = os.getcwd().replace('tests','')
tokenizer = TAPETokenizer(vocab='unirep')
dataset = DM_idp_dataset(cwd=cwd,split='train',tokenizer=tokenizer)

loader = setup_loader(dataset,batch_size=4,local_rank=-1,n_gpu=1,gradient_accumulation_steps=2,num_workers=2)
# # for i,batch in enumerate(loader):
#     print(i)
#     #print(batch)
#     id = batch['input_ids']
#     print(id.shape)
#     mask = batch['input_mask']
#     target = batch['targets']
#     print(id[0],target[0])

import torch
from tape import ProteinBertModel, TAPETokenizer
model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model


sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
token_ids = torch.tensor([tokenizer.encode(sequence)])
output = model(token_ids)
sequence_output = output[0]
# pooled_output = output[1]
print(sequence_output.shape)

import esm

model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
batch_converter = alphabet.get_batch_converter()
print(alphabet.tok_to_idx)
idwsd = alphabet.tok_to_idx
# for idx,k  in enumerate(idwsd):
#     print(f'("{k}",{idx}),')
# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("protein1", " LAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRGIRLLQEE"),

]


batch_labels, batch_strs, batch_tokens = batch_converter(data)
print(batch_strs)
print(batch_tokens)
print(batch_labels)