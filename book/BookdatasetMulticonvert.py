import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import warnings
warnings.filterwarnings('ignore')

def getPath():
    dir = 'BookDatasets/facades/val'
    AB_paths = sorted(makeDataset(dir))
    return AB_paths

def makeDataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if isImageFile(fname):
                path = os.path.join(root, fname)
                img_paths.append(path)

    return img_paths

def isImageFile(fname):
    return fname.endswith('.jpg')


def __transform():
    list = []

    load_size = 286
    list.append(transforms.Resize([load_size, load_size], Image.NEAREST))

    crop_size = 256
    list.append(transforms.Resize([crop_size, crop_size], Image.NEAREST))

    return transforms.Compose(list)

def defaultRGBValue():
    default_RGB_value = [
                        [255,85,0],
                        [255,170,0],
                        [0,85,255],
                        [0,0,255],
                        [0,0,170],
                        [170,255,85],
                        [170,0,0],
                        [0,170,255],
                        [255,255,0],
                        [85,255,170],
                        [255,0,0],
                        [0,255,255],
                        ]
    return default_RGB_value

def calMultichannel(img):
    multi_channel = torch.zeros(12, 256, 256)
    for h in range(256):
        for w in range(256):
            default = defaultRGBValue()
            distance = []
            for index in range(len(default)):
                distance.append(np.linalg.norm(default[index] - img[h][w]))

            min_index = distance.index(min(distance))
            multi_channel[min_index, h, w] = 1

    return multi_channel


def keepPickle(label, index):
    mkdir_name = '/mnt/HDD4TB-3/sugiura/pix2pix/valBookPickle'
    if not os.path.exists(mkdir_name):
        os.mkdir(mkdir_name)
    pickle_name = 'img_numpy' + str(index) + '.pickle'
    pickle_path = os.path.join(mkdir_name, pickle_name)
    print(index)
    with open(pickle_path, mode='wb') as f:
        pickle.dump(label, f)

def main():
    # 学習用データ１つの生成
    # A(テンソル) : 条件画像
    # B(テンソル) : Aのペアとなるターゲット画像

    # ランダムなindexの画像を取得
    AB_paths = getPath()

    for index in range(len(AB_paths)):
        AB_path = AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # 画像を2分割してAとBをそれぞれ取得
        # ランダムシードの生成
        w, h = AB.size
        w2 = int(w / 2)
        # 256x256サイズの画像生成
        # 一度リサイズしてランダムな位置で256x256にcropする
        # AとBは同じ位置からcropする
        transform = __transform()

        #tensor
        label = transform(AB.crop((w2, 0, w, h)))

        label = np.array(label)

        onehot_label = calMultichannel(label)

        keepPickle(onehot_label, index)



if __name__ == '__main__':
    main()