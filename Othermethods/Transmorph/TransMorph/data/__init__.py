import torch.utils.data

def create_dataloader(dataset,batch_size,phase):
    ''' create dataloader '''
    if phase == 'train' :
        return torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers=0,
            pin_memory=True
        )
    elif phase == 'test':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0,
            pin_memory = True
        )
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0,
            pin_memory = True
        )
    else:
        raise NotImplementedError('Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataroot,trainortest,CTorMRI='MRI'):
    if CTorMRI =='CT':
        from data.dataset import CT_Datasets as D
    else:
        from data.dataset import MRI_Datasets as D
    dataset = D(dataroot,trainortest)

    return dataset