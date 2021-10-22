import torch.utils.data as data


def loaders(args, dataset_name=''):
    batch_size = 1
    val_params = {'batch_size' : batch_size,
                  'shuffle'    : False,
                  'num_workers': 2}

    train_params = {'batch_size' : batch_size,
                    'shuffle'    : False,
                    'num_workers': 2}

    if dataset_name == 'DM':
        from dataloaders.dm_loader import DMLoader
        training_set = DMLoader(args, 'train')
        training_generator = data.DataLoader(training_set, **train_params)
        classes = training_set.classes
        val_set = DMLoader(args, 'val')
        val_generator = data.DataLoader(val_set, **val_params)
        classes = [0]
        return training_generator, val_generator,classes

    else:
        from dataloaders.dm_loader import SSLDM
        training_set = SSLDM(args, 'train')
        training_generator = data.DataLoader(training_set, **train_params)
        classes = training_set.classes
        val_set = SSLDM(args, 'val')
        val_generator = data.DataLoader(val_set, **val_params)
        return training_generator, val_generator, classes
