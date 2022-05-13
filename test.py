import generator
import discriminator
import torch.optim as optim
from torch import nn
import torch
import numpy as np
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os
import myPix2pix

def test(model, experiment):
    lossG_list = []
    lossD_list = []

    for index in range(10):
        data_path = '/mnt/HDD4TB-3/sugiura/pix2pix/CMPFacadeDatasets/facades/base'
        file_number = f'{index+1:04}'
        data_fname = 'cmp_b' + file_number + '.jpg'
        data_file_path = os.path.join(data_path, data_fname)
        real = Image.open(data_file_path).convert('RGB')
        load_size = 286
        real = real.resize((load_size, load_size), Image.NEAREST)

        crop_size = 256
        real = real.resize((crop_size, crop_size), Image.NEAREST)

        transform = transforms.ToTensor()
        real = transform(real)

        real = real.to(torch.device("cuda:0"))
        real.unsqueeze_(0)


        pickle_path = '/mnt/HDD4TB-3/sugiura/pix2pix/pickle'
        pickle_fname = 'img_numpy' + str(index) + '.pickle'
        pickle_file_path = os.path.join(pickle_path, pickle_fname)

        with open(pickle_file_path, mode='rb') as f:
            label = pickle.load(f)
        label = label.to(torch.device("cuda:0"))
        label.unsqueeze_(0)

        fake = model.netG(label)
        # Discriminator
        # 条件画像(A)と生成画像(B)を結合
        fakeAB = torch.cat((label, fake), dim=1)
        # 識別器Dに生成画像を入力、このときGは更新しないのでdetachして勾配は計算しない
        pred_fake = model.netD(fakeAB.detach())
        # 偽物画像を入力したときの識別器DのGAN損失を算出
        lossD_fake = model.criterionGAN(pred_fake, False)

        # 条件画像(A)と正解画像(B)を結合
        realAB = torch.cat((label, real), dim=1)
        # 識別器Dに正解画像を入力
        pred_real = model.netD(realAB)
        # 正解画像を入力したときの識別器DのGAN損失を算出
        lossD_real = model.criterionGAN(pred_real, True)

        # 偽物画像と正解画像のGAN損失の合計に0.5を掛ける
        lossD = (lossD_fake + lossD_real) * 0.5

        with torch.no_grad():
            pred_fake = model.netD(fakeAB)

        # 生成器GのGAN損失を算出
        lossG_GAN = model.criterionGAN(pred_fake, True)

        # 生成器Gの損失を合計
        lossG = lossG_GAN

        lossG_list.append(lossG)
        lossD_list.append(lossD)

        output_image = torch.cat([fake, real], dim=3)
        vutils.save_image(output_image,
            '{}/pix2pix_test_{}.png'.format(myPix2pix.Opts().output_dir, index),
            normalize=True)


    meanG = sum(lossG_list) / len(lossG_list)
    meanD = sum(lossD_list) / len(lossD_list)

    f = open("myPix2pixLoss.txt", "w")
    f.write("meanLossG:" + str(meanG))
    f.write("\n")
    f.write("meanLossD:" + str(meanD))
    f.close()