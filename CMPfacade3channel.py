import os.path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class AlignedDataset3CMP(Dataset):
    def __init__(self, config, train_number):
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

        self.train_datasets = []
        for i in range(len(self.A_paths)):
            path = self.A_paths[i]
            number = path[-8] + path[-7] + path[-6] + path[-5]

            if number in train_number:
                self.train_datasets.append(path)

        print(len(self.train_datasets))

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

    def changeFname(self, path):
        fname = path.replace('.jpg', '.png')
        return fname

    def __getitem__(self, index):
        # 学習用データ１つの生成
        # A(テンソル) : 条件画像
        # B(テンソル) : Aのペアとなるターゲット画像
        #-------正解画像の生成-------

        # ランダムなindexの画像を取得
        A_path = self.train_datasets[index]
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
        fname = self.changeFname(A_path)
        B = Image.open(fname).convert('RGB')
        transform = self.__transform()
        #正解画像
        B = transform(B)

        return {'A': B, 'B': A,}

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.train_datasets)



class valAlignedDataset3CMP(Dataset):
    def __init__(self, config, train_number):
        # データセットクラスの初期化
        self.config = config

        dir = os.path.join(config.dataroot, config.phase)
        self.A_paths = sorted(self.__make_dataset_A(dir))

        self.val_datasets = []
        for i in range(len(self.A_paths)):
            path = self.A_paths[i]
            number = path[-8] + path[-7] + path[-6] + path[-5]

            if not number in train_number:
                self.val_datasets.append(path)

        self.val_datasets.append('CMPFacadeDatasets/facades/base/cmp_b0001.jpg')
        print(len(self.val_datasets))


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

    def changeFname(self, path):
        fname = path.replace('.jpg', '.png')
        return fname

    def __getitem__(self, index):
        # 学習用データ１つの生成
        # A(テンソル) : 条件画像
        # B(テンソル) : Aのペアとなるターゲット画像
        #-------正解画像の生成-------

        # ランダムなindexの画像を取得
        A_path = self.val_datasets[index]
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
        fname = self.changeFname(A_path)
        B = Image.open(fname).convert('RGB')
        transform = self.__transform()
        B = transform(B)

        return {'A': B, 'B': A,}

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.val_datasets)