import torch.utils.data as data


def loaders(args, dataset_name=''):
    batch_size = 1
    val_params = {'batch_size' : batch_size,
                  'shuffle'    : False,
                  'num_workers': 2}

    train_params = {'batch_size' : batch_size,
                    'shuffle'    : False,
                    'num_workers': 2}
    from dataloaders.dm_loader import DMLoader
    training_set = DMLoader(args, 'train')
    training_generator = data.DataLoader(training_set, **train_params)

    val_set = DMLoader(args, 'val')
    val_generator = data.DataLoader(training_set, **val_params)
    return training_generator, val_generator
