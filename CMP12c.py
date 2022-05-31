import os.path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pickle
import numpy as np
import random


class AlignedDataset12CMP(Dataset):
    def __init__(self, config, train_number, which):
        # データセットクラスの初期化
        self.config = config
        self.which = which

        # データディレクトリの取得

        #パス名操作に関する処理をまとめたモジュールに実装されている関数の一つです.
        #引数に与えられた二つの文字列を結合させ、一つのパスにする事ができます.
        #二つの間には / が自動で入る
        dir = os.path.join(config.dataroot, config.phase)
        # 画像データパスの取得

        #組み込み関数sorted(): ソートした新たなリストを生成
        #昇順
        #正解画像
        self.A_paths = sorted(self.__make_dataset_A(dir))

        self.train_datasets = []
        self.val_datasets = []

        for i in range(len(self.A_paths)):
            path = self.A_paths[i]
            number = path[-8] + path[-7] + path[-6] + path[-5]

            if number in train_number:
                self.train_datasets.append(path)
            else:
                self.val_datasets.append(path)

        if which == 'train':
            print('train:' + str(len(self.train_datasets)))
        else:
            print('val:' + str(len(self.val_datasets)))



    @classmethod
    def is_image_file_jpg(self, fname):
        # 画像ファイルかどうかを返す
        #fnameに入っているものの末尾がext(ここではpngかjpg)なのかどうかを判断している．
        #一致しているなら，trueを返す．

        #データセットの一枚目だけ強制でバリデーション用画像とする．
        number = fname[-8] + fname[-7] + fname[-6] + fname[-5]
        if number == '0001':
            return False

        return fname.endswith('.jpg')


    @classmethod
    #正解画像データセット作成
    def __make_dataset_A(self, dir):
        # 画像データセットをメモリに格納
        # 画像データセットをメモリに格納
        images_real = []

        #ディレクトリの確認
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        #root:現在のディレクトリ
        # _:内包するディレクトリ
        #fnames:内包するファイル
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file_jpg(fname):
                    path = os.path.join(root, fname)
                    images_real.append(path)

        return images_real


    def __transform(self, param, which):
        list = []

        # 入力画像を一度286x286にリサイズし、その後で256x256
        load_size = self.config.load_size
        list.append(transforms.Resize([load_size, load_size], Image.NEAREST))

        crop_size = self.config.crop_size
        if which == 'train':
            (x, y) = param['crop_pos']
            list.append(transforms.Lambda(lambda img: img.crop((x, y, x + crop_size, y + crop_size))))

            # 1/2の確率で左右反転する
            if param['flip']:
                list.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))
        else:
            list.append(transforms.Resize([crop_size, crop_size], Image.NEAREST))

        # 正解画像は標準化すべき
        # RGB画像をmean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)にNormalizeする
        list += [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(list)


    def __transformPickle(self, param, which):
        list = []
        crop_size = self.config.crop_size

        if which == 'train':
            (x, y) = param['crop_pos']
            list.append(transforms.Lambda(lambda img: img.crop((x, y, x + crop_size, y + crop_size))))

            # 1/2の確率で左右反転する
            if param['flip']:
                list.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))
        else:
            list.append(transforms.Resize([crop_size, crop_size], Image.NEAREST))

        return transforms.Compose(list)


    def __transform_param(self):
        x_max = self.config.load_size - self.config.crop_size
        x = random.randint(0, np.maximum(0, x_max))
        y = random.randint(0, np.maximum(0, x_max))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}


    def pickle_number(self, path):
        pickle_index =  path[-8] + path[-7] + path[-6] + path[-5]
        return pickle_index


    def __getitem__(self, index):
        # 学習用データ１つの生成
        # A(テンソル) : 条件画像
        # B(テンソル) : Aのペアとなるターゲット画像
        #-------正解画像の生成-------

        # ランダムなindexの画像を取得
        if self.which == 'train':
            A_path = self.train_datasets[index]
        else:
            A_path = self.val_datasets[index]
        #Image.open(a):aで指定した画像を開く
        #RGB画像に変換
        A = Image.open(A_path).convert('RGB')

        # 256x256サイズの画像生成
        # 一度リサイズしてランダムな位置で256x256にcropする
        #
        param = self.__transform_param()
        transform = self.__transform(param, self.which)
        #正解画像
        A = transform(A)

        #----------------------------

        #----ラベル画像----
        pickle_index = self.pickle_number(A_path)

        pickle_number = 'CMPfacade286*286Pickle/img_numpy' + pickle_index + '.pickle'
        with open(pickle_number, mode='rb') as f:
            B = pickle.load(f)


        transform_pickle = self.__transformPickle(param, self.which)

        B = transform_pickle(B)

        return {'A': B, 'B': A,}

    def __len__(self):
        # 全画像ファイル数を返す
        if self.which == 'train':
            return len(self.train_datasets)
        else:
            return len(self.val_datasets)