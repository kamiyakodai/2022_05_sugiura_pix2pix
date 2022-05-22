import argparse
import os.path
from Averagemeter import AverageMeter
import torch
from torch.utils.data import DataLoader
import FID
import cometml
import json
import Pix2pixModel
import warnings
warnings.filterwarnings('ignore')
import CMPfacade3channel, CMPfacade12channel
import Averagemeter


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
        self.train_sheets = 400

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
    parser = argparse.ArgumentParser(description='myPx2pix')

    parser.add_argument('-c', '--channels', type=int,
                        choices=[3, 12],
                        help='number of channels.')

    args = parser.parse_args()

    opt = Opts(args)

    if args.channels == 3:
            opt.output_dir = '3ChannelCMPfacadeOutput'
            dataset = CMPfacade3channel.AlignedDataset3CMP(opt)

            val_dataset = CMPfacade3channel.valAlignedDataset3CMP(opt, CMPfacade3channel.AlignedDataset3CMP(opt))
    else:
            opt.output_dir = '12ChannelCMPfacadeOutput'
            dataset = CMPfacade12channel.AlignedDataset12CMP(opt)

            val_dataset = CMPfacade12channel.valAlignedDataset12CMP(opt, CMPfacade12channel.AlignedDataset12CMP(opt))

    model = Pix2pixModel.Pix2Pix(opt)

    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    param_save_path = os.path.join(opt.output_dir, 'param.json')
    save_json(opt.to_dict(), param_save_path, 'w')

    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=True , drop_last = True)

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    experiment = cometml.comet()

    val_lossG = Averagemeter.AverageMeter()
    val_lossD = Averagemeter.AverageMeter()

    """## 学習の開始"""

    for epoch in range(1, opt.epochs + 1):
        model.netG.train()
        model.netD.train()

        for batch_num, data in enumerate(dataloader):
            batches_done = (epoch - 1) * len(dataloader) + batch_num
            model.train(data)
            print(batch_num)

            #len(dataloader)がデータセットによって違うのはなぜ？
            if batch_num % opt.log_interval == 0:
                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    epoch, batch_num, len(dataloader), model.lossD, model.lossG_GAN))

            cometml.gLossComet(experiment, model.lossG_GAN, batches_done)
            cometml.dLossComet(experiment, model.lossD, batches_done)

            if epoch % 100 == 0:
                model.save_image(epoch + 10000)


        for batch_num, data in enumerate(val_loader):
            model.eval(data)

            if batch_num == 0:
                print("===> Validation:Epoch[{}]({}/{})".format(
                    epoch, batch_num, len(val_loader)))

            val_lossG.update(model.lossG_GAN, data['A'].to(torch.device("cuda:0")).shape[0])
            val_lossD.update(model.lossD, data['A'].to(torch.device("cuda:0")).shape[0])

        print("Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    val_lossD.avg, val_lossG.avg))

        cometml.valGLossComet(experiment, model.lossG_GAN, epoch)
        cometml.valDLossComet(experiment, model.lossD_real, epoch)

        if epoch % 5 == 0:
            model.save_model(epoch)
            model.save_image(epoch)

        if epoch % 5 == 0 or epoch == 1 or epoch == 2 or epoch == 3 :
            #FIDscoreの計算
            tmp = val_loader.__iter__()
            data = tmp.next()
            label = data['A'].to(torch.device("cuda:0"))
            real = data['B'].to(torch.device("cuda:0"))
            fake = model.netG(label)
            fretchet_dist = FID.calculate_fretchet(real, fake)
            print(fretchet_dist)
            cometml.FIDComet(experiment, fretchet_dist, epoch)

        model.update_learning_rate()
        val_lossD.reset()
        val_lossG.reset()

if __name__ == '__main__':
    main()