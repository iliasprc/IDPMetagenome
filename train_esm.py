import argparse
import datetime
import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

from dataloaders.dataset import loaders
from models.fair_esm_model import IDP_esm1_t6_43M_UR50S, IDP_esm1_t12_85M_UR50S, IDP_esm1_msa
from trainer.logger import Logger
from trainer.util import reproducibility, select_optimizer, load_checkpoint, _parse_args, arguments

config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                           help='YAML config file specifying default arguments')


def main():
    args = arguments()
    args, args_text = _parse_args(config_parser, args)
    now = datetime.datetime.now()
    cwd = os.getcwd()
    # if len(myargs) > 0:
    #     if 'c' in myargs:
    #         config_file = myargs['c']
    # else:
    #     config_file = './config/esm_config.yaml'
    #
    # config = OmegaConf.load(os.path.join(cwd, config_file))['trainer']
    # config.cwd = str(cwd)
    reproducibility(args)
    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(args.cwd,
                                 f'checkpoints/dataset_{args.dataset.name}/model_{args.model.name}/date_'
                                 f'{dt_string}')

    log = Logger(path=cpkt_fol_name, name='LOG').get_logger()

    log.info(f"Checkpoint folder {cpkt_fol_name}")

    writer = SummaryWriter(cpkt_fol_name + '/runs/')

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f'device: {device}')

    training_generator, val_generator, test_gen, classes = loaders(args=args, dataset_name=args.dataset )
    log.info(f'train {len(training_generator)} dev {len(val_generator)} test ')

    if args.model == 'IDP_esm1_t12_85M_UR50S':
        model = IDP_esm1_t12_85M_UR50S()
    elif args.model == 'IDP_esm1_t6_43M_UR50S':
        model = IDP_esm1_t6_43M_UR50S()
    elif args.model.name == 'IDP_esm1_msa':
        model = IDP_esm1_msa()

    log.info(f"{model}")
    if (args.load):
        pth_file, _ = load_checkpoint(args.pretrained_cpkt, model, strict=True, load_seperate_layers=False)

    if (args.cuda and use_cuda):
        if torch.cuda.device_count() > 1:
            log.info(f"Let's use {torch.cuda.device_count()} GPUs!")

            model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer, scheduler = select_optimizer(model, args['model'], None)

    log.info(f"Checkpoint Folder {cpkt_fol_name} ")
    shutil.copy(os.path.join(args.cwd, config_file), cpkt_fol_name)

    from trainer.esm_trainer import ESMTrainer
    trainer = ESMTrainer(args, model=model, optimizer=optimizer,
                         data_loader=training_generator, writer=writer, logger=log,
                         valid_data_loader=val_generator, test_data_loader=test_gen, class_dict=classes,
                         lr_scheduler=scheduler,
                         checkpoint_dir=cpkt_fol_name)
    trainer.train()


if __name__ == '__main__':
    main()
