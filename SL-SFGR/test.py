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

if __name__ == "__main__":
    torch.cuda.set_device(5)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/diffuseMorph_test_3D.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='5,6,7')
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
            test_set = data.create_dataset(opt['datasets']['test']['dataroot'],trainortest=phase)
            test_loader = data.create_dataloader(test_set, batchSize, phase)
    logger.info('Initial dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')


    #Test Model
    registTime = []
    logger.info('Begin Model Evaluation.')
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path,exist_ok=True)

    for istep,test_data in enumerate(test_loader):
        idx+=1
        dataname=str(test_data['name_m'][0][-24:-7])+'_to_'+str(test_data['name_f'][0][-24:-7])
        diffusion.feed_data(test_data)
        final_flow,warped,warped_mask,seg_out_m,seg_out_f = diffusion.test_registration()

        final_flow = final_flow.squeeze().cpu().numpy()
        warped = warped.squeeze().cpu().numpy()
        warped_mask = warped_mask.int().squeeze().cpu().numpy()
        seg_out_m = seg_out_m.squeeze().cpu().numpy()
        seg_out_f = seg_out_f.squeeze().cpu().numpy()

        seg_out_f = np.argmax(seg_out_f,axis=0).astype(np.float32)
        seg_out_m = np.argmax(seg_out_m,axis=0).astype(np.float32)

        final_flow = np.transpose(final_flow, (1, 2, 3, 0))

        sample_img = sitk.ReadImage(test_data['name_f'][0])
        origin = sample_img.GetOrigin()
        direction = sample_img.GetDirection()
        spacing = sample_img.GetSpacing()

        final_flow = sitk.GetImageFromArray(final_flow)
        warped = sitk.GetImageFromArray(warped)
        warped_mask = sitk.GetImageFromArray(warped_mask)
        seg_out_m = sitk.GetImageFromArray(seg_out_m)
        seg_out_f = sitk.GetImageFromArray(seg_out_f)

        final_flow.SetDirection(direction)
        final_flow.SetOrigin(origin)
        final_flow.SetSpacing(spacing)

        warped.SetDirection(direction)
        warped.SetOrigin(origin)
        warped.SetSpacing(spacing)

        warped_mask.SetDirection(direction)
        warped_mask.SetOrigin(origin)
        warped_mask.SetSpacing(spacing)

        seg_out_m.SetDirection(direction)
        seg_out_m.SetOrigin(origin)
        seg_out_m.SetSpacing(spacing)

        seg_out_f.SetDirection(direction)
        seg_out_f.SetOrigin(origin)
        seg_out_f.SetSpacing(spacing)

        sitk.WriteImage( final_flow,result_path+dataname+'_flow.nii.gz')
        sitk.WriteImage( warped,result_path+dataname+'_warped.nii.gz')
        sitk.WriteImage( warped_mask,result_path+dataname+'_warped_mask.nii.gz')
        sitk.WriteImage( seg_out_m,result_path+dataname+'_seg_out_m.nii.gz')
        sitk.WriteImage( seg_out_f,result_path+dataname+'_seg_out_f.nii.gz')
    


