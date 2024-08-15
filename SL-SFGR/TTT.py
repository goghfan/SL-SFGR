import torch
import data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
from math import *
import time
from util.visualizer import Visualizer
from PIL import Image
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk
def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

if __name__ == "__main__":
    torch.cuda.set_device(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/diffuseMorph_train_3D.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='4,5,6,7')
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase != opt['phase']: continue
        if opt['phase'] == 'train':
            batchSize = opt['datasets']['train']['batch_size']
            train_set = data.create_dataset(opt['datasets']['train']['dataroot'],trainortest=phase)
            train_loader = data.create_dataloader(train_set, batchSize, phase)
            # if torch.cuda.is_available() and torch.cuda.device_count()>1:
            #     train_loader = torch.nn.DataParallel(train_loader)
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
        elif opt['phase'] == 'test':
            batchSize = 1
            test_set = data.create_dataset(opt['datasets']['test']['dataroot'],type='nii.gz',trainortest=phase)
            test_loader = data.create_dataloader(test_set, batchSize, phase)
    logger.info('Initial dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    if opt['phase'] == 'train':
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        n_epoch = opt['train']['n_epoch']
        if opt['path']['resume_state']:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

        while current_epoch < n_epoch:
            current_epoch += 1
            for istep, train_data in enumerate(train_loader):
                iter_start_time = time.time()
                current_step += 1
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if (istep + 1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    visualizer.print_current_errors(current_epoch, istep + 1, training_iters, logs, t, 'Train')
                    visualizer.plot_current_errors(current_epoch, (istep + 1) / float(training_iters), logs)
            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        pass