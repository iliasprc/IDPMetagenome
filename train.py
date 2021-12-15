import datetime
import os
import shutil

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloaders.dataset import loaders
from trainer.logger import Logger
from trainer.util import reproducibility, select_model, select_optimizer, load_checkpoint, get_arguments,_parse_args,arguments

config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                           help='YAML config file specifying default arguments')

def main():
    args =  arguments()
    args, args_text = _parse_args(config_parser,args)
    myargs = []  # getopts(sys.argv)
    cwd = os.getcwd()
    now = datetime.datetime.now()
    config_file = './config/config_parse.yaml'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    #config = OmegaConf.load(os.path.join(cwd, config_file))


    args.cwd = str(cwd)
    reproducibility(args)
    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(args.cwd,
                                 f'checkpoints/dataset_{args.dataset}/model_{args.model}/date_'
                                 f'{dt_string}')

    log = Logger(path=cpkt_fol_name, name='LOG').get_logger()

    log.info(f"Checkpoint folder {cpkt_fol_name}")

    writer = SummaryWriter(cpkt_fol_name + '/runs/')

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f'device: {device}')

    training_generator, val_generator, test_generator,classes = loaders(args=args, dataset_name=args.dataset)
    n_classes = len(classes)

    # if args.dataset.name == 'celeba
    # n_classes = 1
    model = select_model(args, 2)
    # model.head = torch.nn.Linear(128,20)
    log.info(f"{model}")

    if (args.load):
        # model.head = torch.nn.Linear(512, n_classes)
        model.embed = torch.nn.Embedding(22, 128)
        # model.head = torch.nn.Linear(128, 22)
        model.head = torch.nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 22))
        pth_file, _ = load_checkpoint(args.pretrained_cpkt, model, strict=True, load_seperate_layers=False)
        model.head = torch.nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 2))



    else:
        pth_file = None
    if (args.cuda and use_cuda):
        if torch.cuda.device_count() > 1:
            log.info(f"Let's use {torch.cuda.device_count()} GPUs!")

            model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer, scheduler = select_optimizer(model, args, None)
    # log.info(f'{model}')
    log.info(f"Checkpoint Folder {cpkt_fol_name} ")
    shutil.copy(os.path.join(args.cwd, config_file), cpkt_fol_name)

    from trainer.trainer import Trainer
    trainer = Trainer(args, model=model, optimizer=optimizer,
                      data_loader=training_generator, writer=writer, logger=log,
                      valid_data_loader=val_generator, class_dict=classes,
                      lr_scheduler=scheduler,
                      checkpoint_dir=cpkt_fol_name)
    trainer.train()


if __name__ == '__main__':
    main()
