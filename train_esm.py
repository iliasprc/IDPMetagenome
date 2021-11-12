import datetime
import os
import shutil

import esm
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from dataloaders.dataset import loaders
from models.fair_esm_model import IDP_esm1_t6_43M_UR50S, IDP_esm1_t12_85M_UR50S
from trainer.logger import Logger
from trainer.util import load_checkpoint
from trainer.util import reproducibility, select_optimizer, get_arguments

#
# Load ESM-1b model
model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
batch_converter = alphabet.get_batch_converter()


# ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
# ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
# ("protein3",  "K A <mask> I S Q"),
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#
# ]
# batch_labels, batch_strs, batch_tokens = batch_converter(data)
# print(batch_tokens)
# # Extract per-residue representations (on CPU)
# with torch.no_grad():
#     results = model(batch_tokens, repr_layers=[6], return_contacts=True)
# print(results.keys())
# #print(model)
# token_representations = results["representations"][6]
# print(len(results["representations"]),token_representations.shape)
# exit()
def main():
    args = get_arguments()
    myargs = []  # getopts(sys.argv)
    now = datetime.datetime.now()
    cwd = os.getcwd()
    if len(myargs) > 0:
        if 'c' in myargs:
            config_file = myargs['c']
    else:
        config_file = 'config/esm_config.yaml'

    config = OmegaConf.load(os.path.join(cwd, config_file))['trainer']
    config.cwd = str(cwd)
    reproducibility(config)
    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(config.cwd,
                                 f'checkpoints/dataset_{config.dataset.name}/model_{config.model.name}/date_'
                                 f'{dt_string}')

    log = Logger(path=cpkt_fol_name, name='LOG').get_logger()

    log.info(f"Checkpoint folder {cpkt_fol_name}")
    log.info(f"date and time = {dt_string}")

    log.info(f'pyTorch VERSION:{torch.__version__}', )
    log.info(f'CUDA VERSION')

    log.info(f'CUDNN VERSION:{torch.backends.cudnn.version()}')
    log.info(f'Number CUDA Devices: {torch.cuda.device_count()}')

    if args.tensorboard:

        # writer_path = os.path.join(config.save,
        #                            'checkpoints/model_' + config.model.name + '/dataset_' + config.dataset.name +
        #                            '/date_' + dt_string + '/runs/')

        writer = SummaryWriter(cpkt_fol_name + '/runs/')
    else:
        writer = None

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f'device: {device}')

    training_generator, val_generator, classes = loaders(args=config, dataset_name=config.dataset.name)
    n_classes = len(classes)

    if config.model.name == 'IDP_esm1_t12_85M_UR50S':
        model = IDP_esm1_t12_85M_UR50S()
    elif config.model.name == 'IDP_esm1_t6_43M_UR50S':
        model = IDP_esm1_t6_43M_UR50S()
    # model.head = torch.nn.Linear(128,20)
    log.info(f"{model}")
    if (config.load):

        pth_file, _ = load_checkpoint(config.pretrained_cpkt, model, strict=True, load_seperate_layers=False)
        model.fc = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.5), nn.Linear(768, 2))
    if (config.cuda and use_cuda):
        if torch.cuda.device_count() > 1:
            log.info(f"Let's use {torch.cuda.device_count()} GPUs!")

            model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer, scheduler = select_optimizer(model, config['model'], None)
    # log.info(f'{model}')
    log.info(f"Checkpoint Folder {cpkt_fol_name} ")
    shutil.copy(os.path.join(config.cwd, config_file), cpkt_fol_name)

    from trainer.esm_trainer import ESMTrainer
    trainer = ESMTrainer(config, model=model, optimizer=optimizer,
                         data_loader=training_generator, writer=writer, logger=log,
                         valid_data_loader=val_generator, class_dict=classes,
                         lr_scheduler=scheduler,
                         checkpoint_dir=cpkt_fol_name)
    trainer.train()


if __name__ == '__main__':
    main()
