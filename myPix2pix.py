import sys
import argparse
import os.path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import torchvision.models as models
from torch.nn import functional as FF
import FID
import cometml
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings('ignore')
import json
import Pix2pixModel
import getDataset


def save_json(file, param_save_path, mode):
    with open(param_save_path, mode) as outfile:
        json.dump(file, outfile, indent=4)


class Opts():
    def __init__(self, args):
        self.epochs = 200
        self.save_data_interval = 10
        self.save_image_interval = 10
        self.log_interval = 20
        self.sample_interval = 10
        self.batch_size = 64
        self.load_size = 286
        self.crop_size = 256
        self.cpu = True
        self.dataroot = 'CMPFacadeDatasets/facades'
        self.output_dir = 'myPix2pixOutput'
        self.log_dir = './logs'
        self.phase = 'base'
        self.lambda_L1 = 100.0
        self.epochs_lr_decay = 0
        self.epochs_lr_decay_start = -1
        self.path_to_generator = None
        self.path_to_discriminator = None
        self.device_name = "cuda:0"
        self.device = torch.device(self.device_name)
        self.input_channel = args.channels

    def to_dict(self):
        parameters = {
            'epochs': self.epochs,
            'save_data_interval': self.save_data_interval,
            'save_image_interval': self.save_image_interval,
            'log_interval': self.log_interval,
            'sample_interval': self.sample_interval,
            'batch_size': self.batch_size,
            'load_size': self.load_size,
            'crop_size': self.crop_size,
            'cpu': self.cpu,
            'dataroot': self.dataroot,
            'output_dir': self.output_dir,
            'log_dir': self.log_dir,
            'phase': self.phase,
            'lambda_L1': self.lambda_L1,
            'epochs_lr_decay': self.epochs_lr_decay,
            'epochs_lr_decay_start': self.epochs_lr_decay_start,
            'path_to_generator': self.path_to_generator,
            'path_to_discriminator': self.path_to_discriminator,
            'device_name': self.device_name,
        }
        return parameters


def main():
    parser = argparse.ArgumentParser(description='simple CNN model')

    parser.add_argument('-c', '--channels', type=int,
                        choices=[3, 12],
                        help='number of channels.')

    parser.add_argument('-d', '--dataset', type=str,
                        choices=['book', 'CMPfacade'],
                        help='dataset name')


    args = parser.parse_args()

    opt = Opts(args)

    model = Pix2pixModel.Pix2Pix(opt)

    if args.channels == 3:
        if args.dataset == 'book':
            dataset = getDataset.AlignedDataset3Book(opt)
            opt.output_dir = '3ChannelBookOutput'
        else:
            dataset = getDataset.AlignedDataset3CMP(opt)
            opt.output_dir = '3ChannelCMPfacadeOutput'
    else:
        if args.dataset == 'book':
            dataset = getDataset.AlignedDataset12Book(opt)
            opt.output_dir = '12ChannelBookOutput'
        else:
            dataset = getDataset.AlignedDataset12CMP(opt)
            opt.output_dir = '12ChannelCMPfacadeOutput'

    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    param_save_path = os.path.join(opt.output_dir, 'param.json')
    save_json(opt.to_dict(), param_save_path, 'w')

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    experiment = cometml.comet()

    """## 学習の開始"""

    for epoch in range(1, opt.epochs + 1):

        for batch_num, data in enumerate(dataloader):
            model.netG.train()
            model.netD.train()
            model.train(data)

            #len(dataloader)がデータセットによって違うのはなぜ？
            if batch_num % opt.log_interval == 0:
                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    epoch, batch_num, len(dataloader), model.lossD_real, model.lossG_GAN))
                cometml.gLossComet(experiment, model.lossG_GAN, epoch)
                cometml.dLossComet(experiment, model.lossD_real, epoch)

        if epoch % opt.save_data_interval == 0:
            for batch_num, data in enumerate(val_loader):
                model.netG.eval()
                model.netD.eval()
                model.eval(data)

                if batch_num % opt.log_interval == 0:
                    print("===> Validation:Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                        epoch, batch_num, len(dataloader), model.lossD_real, model.lossG_GAN))
                    cometml.valGLossComet(experiment, model.lossG_GAN, epoch)
                    cometml.valDLossComet(experiment, model.lossD_real, epoch)

        if epoch % opt.save_data_interval == 0:
            model.save_model(epoch)

        if epoch % opt.save_image_interval == 0:
            model.save_image(epoch)

        #FIDscoreの計算
        tmp = dataloader.__iter__()
        data = tmp.next()
        label = data['A'].to(torch.device("cuda:0"))
        real = data['B'].to(torch.device("cuda:0"))
        fake = model.netG(label)
        fretchet_dist = FID.calculate_fretchet(real, fake)
        print(fretchet_dist)

        if epoch % 10 == 0 or epoch == 1 or epoch == 2 or epoch == 3 :
            cometml.FIDComet(experiment, fretchet_dist, epoch)

        model.update_learning_rate()

if __name__ == '__main__':
    main()