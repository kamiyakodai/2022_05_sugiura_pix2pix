# 条件画像と正解画像のペアデータセット生成クラス
import os.path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Dataset
import pickle


#-----３チャンネルBookDatasets-------------------------------------
class AlignedDataset3Book(Dataset):
    IMG_EXTENSIONS = ['.png', 'jpg']
    # configは全ての学習条件を格納する

    # 画像データは'/path/to/data/train'および'/path/to/data/test'に
    # {A,B}の形式で格納されているものとみなす

    def __init__(self, config):
        # データセットクラスの初期化
        self.config = config

        # データディレクトリの取得
        dir = os.path.join(config.dataroot, config.phase)
        print(dir)
        # 画像データパスの取得
        self.AB_paths = sorted(self.__make_dataset(dir))

    @classmethod
    def is_image_file(self, fname):
        # 画像ファイルかどうかを返す
        return any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)

    @classmethod
    def __make_dataset(self, dir):
        # 画像データセットをメモリに格納
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __transform(self):
        list = []

        load_size = self.config.load_size
        # 入力画像を一度286x286にリサイズし、その後で256x256
        list.append(transforms.Resize([load_size, load_size], Image.NEAREST))

        crop_size = self.config.crop_size
        list.append(transforms.Resize([crop_size, crop_size], Image.NEAREST))

        # RGB画像をmean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)にNormalizeする
        list += [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(list)


    def __getitem__(self, index):
        # 学習用データ１つの生成
        # A(テンソル) : 条件画像
        # B(テンソル) : Aのペアとなるターゲット画像

        # ランダムなindexの画像を取得
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # 画像を2分割してAとBをそれぞれ取得
        # ランダムシードの生成
        w, h = AB.size
        w2 = int(w / 2)
        # 256x256サイズの画像生成
        # 一度リサイズしてランダムな位置で256x256にcropする
        # AとBは同じ位置からcropする
        transform = self.__transform()
        A = transform(AB.crop((0, 0, w2, h)))
        B = transform(AB.crop((w2, 0, w, h)))

        #return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
        return {'A': B, 'B': A, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.AB_paths)



#-----３チャンネルCMPfacadeDatasets-------------------------------------

class AlignedDataset3CMP(Dataset):
    # configは全ての学習条件を格納する

    # 画像データは'/path/to/data/train'および'/path/to/data/test'に
    # {A,B}の形式で格納されているものとみなす

    def __init__(self, config):
        # データセットクラスの初期化
        self.config = config

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
        #ラベル画像
        self.B_paths = sorted(self.__make_dataset_B(dir))

    @classmethod
    def is_image_file_png(self, fname):
        # 画像ファイルかどうかを返す
        #fnameに入っているものの末尾がext(ここではpngかjpg)なのかどうかを判断している．
        #一致しているなら，trueを返す．
        return fname.endswith('.png')
    @classmethod
    def is_image_file_jpg(self, fname):
        # 画像ファイルかどうかを返す
        #fnameに入っているものの末尾がext(ここではpngかjpg)なのかどうかを判断している．
        #一致しているなら，trueを返す．
        return fname.endswith('jpg')

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

    #ラベル画像データセット作成
    @classmethod
    def __make_dataset_B(self, dir):

        images_label = []

        #ディレクトリの確認
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        #root:現在のディレクトリ
        # _:内包するディレクトリ
        #fnames:内包するファイル
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file_png(fname):
                    path = os.path.join(root, fname)
                    images_label.append(path)

        return images_label


    def __transform(self):
        list = []

        # 入力画像を一度286x286にリサイズし、その後で256x256
        load_size = self.config.load_size
        list.append(transforms.Resize([load_size, load_size], Image.NEAREST))

        crop_size = self.config.crop_size
        list.append(transforms.Resize([crop_size, crop_size], Image.NEAREST))

        # RGB画像をmean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)にNormalizeする
        list += [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(list)

    def __getitem__(self, index):
        # 学習用データ１つの生成
        # A(テンソル) : 条件画像
        # B(テンソル) : Aのペアとなるターゲット画像

        # ランダムなindexの画像を取得
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        #Image.open(a):aで指定した画像を開く
        #RGB画像に変換
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # 256x256サイズの画像生成
        # 一度リサイズしてランダムな位置で256x256にcropする
        # AとBは同じ位置からcropする
        transform = self.__transform()
        #正解画像
        A = transform(A)
        #ラベル画像(セグメンテーション)
        B = transform(B)

        #return {'A': A, 'B': B}
        return {'A': B, 'B': A}

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.A_paths)



#-----12チャンネルBookDatasets-----------------------------------------
class AlignedDataset12Book(Dataset):
    IMG_EXTENSIONS = ['.png', 'jpg']
    # configは全ての学習条件を格納する

    # 画像データは'/path/to/data/train'および'/path/to/data/test'に
    # {A,B}の形式で格納されているものとみなす

    def __init__(self, config):
        # データセットクラスの初期化
        self.config = config

        # データディレクトリの取得
        dir = os.path.join(config.dataroot, config.phase)
        print(dir)
        # 画像データパスの取得
        self.AB_paths = sorted(self.__make_dataset(dir))

    @classmethod
    def is_image_file(self, fname):
        # 画像ファイルかどうかを返す
        return any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)

    @classmethod
    def __make_dataset(self, dir):
        # 画像データセットをメモリに格納
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __transform(self):
        list = []

        load_size = self.config.load_size
        # 入力画像を一度286x286にリサイズし、その後で256x256
        list.append(transforms.Resize([load_size, load_size], Image.NEAREST))

        crop_size = self.config.crop_size
        list.append(transforms.Resize([crop_size, crop_size], Image.NEAREST))

        # RGB画像をmean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)にNormalizeする
        list += [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(list)


    def __getitem__(self, index):
        # 学習用データ１つの生成
        # A(テンソル) : 条件画像
        # B(テンソル) : Aのペアとなるターゲット画像

        # ランダムなindexの画像を取得
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # 画像を2分割してAとBをそれぞれ取得
        # ランダムシードの生成
        w, h = AB.size
        w2 = int(w / 2)
        # 256x256サイズの画像生成
        # 一度リサイズしてランダムな位置で256x256にcropする
        # AとBは同じ位置からcropする
        transform = self.__transform()
        A = transform(AB.crop((0, 0, w2, h)))

        pickle_number = "BookPickle/img_numpy" + str(index) + ".pickle"
        with open(pickle_number, mode='rb') as f:
            B = pickle.load(f)

        #return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
        return {'A': B, 'B': A, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.AB_paths)



#-----１２チャンネルCMPfacadeDatasets-------------------------------------

class AlignedDataset12CMP(Dataset):
    # configは全ての学習条件を格納する

    # 画像データは'/path/to/data/train'および'/path/to/data/test'に
    # {A,B}の形式で格納されているものとみなす

    def __init__(self, config):
        # データセットクラスの初期化
        self.config = config

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

    @classmethod
    def is_image_file_jpg(self, fname):
        # 画像ファイルかどうかを返す
        #fnameに入っているものの末尾がext(ここではpngかjpg)なのかどうかを判断している．
        #一致しているなら，trueを返す．
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


    def __transform(self):
        list = []

        # 入力画像を一度286x286にリサイズし、その後で256x256
        load_size = self.config.load_size
        list.append(transforms.Resize([load_size, load_size], Image.NEAREST))

        crop_size = self.config.crop_size
        list.append(transforms.Resize([crop_size, crop_size], Image.NEAREST))

        # 正解画像は標準化すべき
        # RGB画像をmean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)にNormalizeする
        list += [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(list)

    def __getitem__(self, index):
        # 学習用データ１つの生成
        # A(テンソル) : 条件画像
        # B(テンソル) : Aのペアとなるターゲット画像
        #-------正解画像の生成-------

        # ランダムなindexの画像を取得
        A_path = self.A_paths[index]
        #Image.open(a):aで指定した画像を開く
        #RGB画像に変換
        A = Image.open(A_path).convert('RGB')

        # 256x256サイズの画像生成
        # 一度リサイズしてランダムな位置で256x256にcropする
        # AとBは同じ位置からcropする
        transform = self.__transform()
        #正解画像
        A = transform(A)

        #----------------------------

        #----ラベル画像----

        pickle_number = "CMPfacadePickle/img_numpy" + str(index) + ".pickle"
        with open(pickle_number, mode='rb') as f:
            B = pickle.load(f)

        return {'A': B, 'B': A}

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.A_paths)
