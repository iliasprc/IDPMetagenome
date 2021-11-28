# username, password = 'papastrai@csd.auth.gr', 'vyiNvYCD6g2v6B9'
#
# # from pangea_api import Knex, User, Organization
# #
# # knex = Knex()
# # User(knex, "papastrai@csd.auth.gr", password).login()
# # org = Organization(knex, "MetaSUB Consortium").idem()
# # grp = org.sample_group("MetaSUB Doha").idem()
# # for sample in grp.get_samples(cache=False):
# #     for ar in sample.get_analysis_results(cache=False):
# #         if ar.module_name != 'raw::raw_reads':
# #             continue
# #         for field in ar.get_fields(cache=False):
# #             field.download_file(filename=filename)
#
# # pangea-api download sample-results -e 'papastrai@csd.auth.gr' -p 'vyiNvYCD6g2v6B9' --module-name "raw::raw_reads"
# # 'MetaSUB Consortium' 'MetaSUB Doha'
# # pangea-api download sample-results -e 'papastrai@csd.auth.gr' -p vyiNvYCD6g2v6B9 --module-name 'raw::raw_reads' 'MetaSUB Consortium' 'MetaSUB Doha'
# # pangea-api download metadata -e 'papastrai@csd.auth.gr' -p vyiNvYCD6g2v6B9 'MetaSUB Consortium' 'MetaSUB Doha'
#
# pangea-api download sample-results -e 'papastrai@csd.auth.gr' -p vyiNvYCD6g2v6B9 --module-name "raw::raw_reads" "MetaSUB Consortium" "MetaSUB Doha"
# import pandas as pd
# from pangea_api import Knex, User, Organization
#
# knex = Knex()
# User(knex, "papastrai@csd.auth.gr", password).login()
# print('sdsds')
# org = Organization(knex, "MetaSUB Consortium").idem()
# grp = org.sample_group("MetaSUB Doha").idem()
#
# for sample in grp.get_samples(cache=False):
#     print(sample)
#     if sample_names and sample.name not in sample_names:
#         continue
#     metadata[sample.name] = sample.metadata
# metadata = pd.DataFrame.from_dict(metadata, orient='index')
# metadata.to_csv("MetaSUB Doha_metadata.csv")
# fpath = '/home/iliask/PycharmProjects/uniref50.fasta'
# from pysam import FastaFile
# sequences_object = FastaFile(fpath)
#
# from allennlp.commands.elmo import ElmoEmbedder
# from pathlib import Path
#
# model_dir = Path('/config/uniref50_v2')
# weights = model_dir / 'weights.hdf5'
# options = model_dir / 'options.json'
# embedder = ElmoEmbedder(options,weights, cuda_device=0)
#
#
# seq = 'MEGSKTSNNSTMQVSFVCQRCSQPLKLDTSFKILDRVTIQELTAPLLTTAQAKPGETQEEETNSGEEPFIETPRQDGVSRRFIPPARMMSTESANSFTLIGEASDGGTMENLSRRLKVTGDLFDIMSGQTDVDHPLCEECTDTLLDQLDTQLNVTENECQNYKRCLEILEQMNEDDSEQLQMELKELALEEERLIQELEDVEKNRKIVAENLEKVQAEAERLDQEEAQYQREYSEFKRQQLELDDELKSVENQMRYAQTQLDKLKKTNVFNATFHIWHSGQFGTINNFRLGRLPSVPVEWNEINAAWGQTVLLLHALANKMGLKFQRYRLVPYGNHSYLESLTDKSKELPLYCSGGLRFFWDNKFDHAMVAFLDCVQQFKEEVEKGETRFCLPYRMDVEKGKIEDTGGSGGSYSIKTQFNSEEQWTKALKFMLTNLK' \
#       'WGLAWVSSQFYNK'
# embedding = embedder.embed_sentence(list(seq)) #
# for i in range(1000):
#       embedding = embedder.embed_sentence(list(seq))
#       print(embedding.shape)
# print(embedder)


import torch
import os
from torch.utils.data import DataLoader, RandomSampler, Dataset
from dataloaders.utils._sampler import BucketBatchSampler
from dataloaders.idp_loaderv2 import DM_idp_dataset
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
for i,batch in enumerate(loader):
    print(i)
    print(len(batch))
    ids = batch['input_ids']
    #mask = batch['mask']

