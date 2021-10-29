import torch.utils.data as data


def loaders(args, dataset_name=''):
    classes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
               'Y']
    batch_size = 1
    val_params = {'batch_size' : batch_size,
                  'shuffle'    : False,
                  'num_workers': 2}

    train_params = {'batch_size' : batch_size,
                    'shuffle'    : True,
                    'num_workers': 2}

    if dataset_name == 'DM':
        from dataloaders.dm_loader import DMLoader
        training_set = DMLoader(args, 'train')
        training_generator = data.DataLoader(training_set, **train_params)
        classes = training_set.classes
        val_set = DMLoader(args, 'val')
        val_generator = data.DataLoader(val_set, **val_params)

        return training_generator, val_generator,classes

    elif dataset_name == 'MXD494':
        from dataloaders.dm_loader import MXD494Loader
        training_set = MXD494Loader(args, 'train')
        training_generator = data.DataLoader(training_set, **train_params)
        classes = training_set.classes
        val_set = MXD494Loader(args, 'val')
        val_generator = data.DataLoader(val_set, **val_params)


        return training_generator, val_generator, classes
    # elif dataset_name == 'SSLDM':
    #     from dataloaders.dm_loader import SSLDM
    #     training_set = SSLDM(args, 'train')
    #     training_generator = data.DataLoader(training_set, **train_params)
    #     classes = training_set.classes
    #     val_set = SSLDM(args, 'val')
    #     val_generator = data.DataLoader(val_set, **val_params)
    #     return training_generator, val_generator, classes