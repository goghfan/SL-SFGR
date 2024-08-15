from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
# from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import data as datass
from math import ceil
device = 'cuda:0'

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def set_device(device, x):
    if isinstance(x, dict):
        for key, item in x.items():
            if item is not None and not isinstance(item, list):
                x[key] = item.to(device)
    elif isinstance(x, list):
        for item in x:
            if item is not None:
                item = item.to(device)
    else:
        x = x.to(device)
    return x

def main():
    train_dir = '/root/autodl-tmp/'
    val_dir = '/root/autodl-tmp/'
    weights = [1, 0.02] # loss weights
    save_dir = '/home/vrdoc/GF/lung_registration/BaseLine/registration/Transmorph/TransMorph_Transformer_for_Medical_Image_Registration/TransMorph/experiments/TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 300 #max traning epoch
    cont_training = False
     #if continue training

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model((192,192,160), 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model((192,192,160), 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 50
        model_dir = '/root/autodl-tmp/BIBM2024/Transmorph/TransMorph_Transformer_for_Medical_Image_Registration/TransMorph/experiments/TransMorph_mse_1_diffusion_0.02/'
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    phase = 'train'
    if phase == 'train':
        batchSize = 2
        train_set = datass.create_dataset(train_dir,trainortest=phase)
        train_loader = datass.create_dataloader(train_set, batchSize, phase)
                # if torch.cuda.is_available() and torch.cuda.device_count()>1:
                #     train_loader = torch.nn.DataParallel(train_loader)
        training_iters = int(ceil(train_set.data_len / float(batchSize)))
        val_set = datass.create_dataset(val_dir,trainortest='validation')        
        val_loader = datass.create_dataloader(val_set, batchSize, phase)
    else:
        batchSize = 2
        train_set = datass.create_dataset(train_dir,trainortest=phase)
        train_loader = datass.create_dataloader(train_set, batchSize, phase)
                # if torch.cuda.is_available() and torch.cuda.device_count()>1:
                #     train_loader = torch.nn.DataParallel(train_loader)
        training_iters = int(ceil(train_set.data_len / float(batchSize)))
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            torch.cuda.empty_cache()
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = set_device(device,data)
            x = data['M']
            y = data['F']
            x_in = torch.cat((x,y), dim=1)
            output = model(x_in)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del x_in
            del output

            # # flip fixed and moving images 双向的，不用双向了
            # loss = 0
            # x_in = torch.cat((y, x), dim=1)
            # output = model(x_in)
            # for n, loss_function in enumerate(criterions):
            #     curr_loss = loss_function(output[n], x) * weights[n]
            #     loss_vals[n] += curr_loss
            #     loss += curr_loss
            # loss_all.update(loss.item(), y.numel())
            # # compute gradient and do SGD step
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # torch.cuda.empty_cache()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        model.eval()
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                data = set_device(device,data)
                x = data['M']
                y = data['F']
                x_seg = data['ML']
                y_seg = data['FL']
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, (192,192,160)).squeeze(0)
                output = model(x_in)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg.long(), 46)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.unsqueeze(0).detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(192,192,160)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()