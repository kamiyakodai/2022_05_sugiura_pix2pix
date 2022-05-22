from PIL import Image
import os
import torchvision.utils as vutils

def is_image_file(fname):
    IMG_EXTENSIONS = ['.png', 'jpg']
    # 画像ファイルかどうかを返す
    #fnameに入っているものの末尾がext(ここではpngかjpg)なのかどうかを判断している．
    #一致しているなら，trueを返す．
    return any(fname.endswith(ext) for ext in IMG_EXTENSIONS)

#正解画像データセット作成
def makeDataset(dir):
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
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images_real.append(path)

    return images_real

def main():
    train_dir = '/mnt/HDD4TB-3/sugiura/pix2pix/CMPFacadeDatasets/facades/train'
    val_dir = '/mnt/HDD4TB-3/sugiura/pix2pix/CMPFacadeDatasets/facades/val'
    base_dir = '/mnt/HDD4TB-3/sugiura/pix2pix/CMPFacadeDatasets/facades/base'

    train_path = sorted(makeDataset(train_dir))
    train_end = train_path[-1]
    number = train_end[-8] + train_end[-7] + train_end[-6] + train_end[-5]
    number = int(number)
    val_path = sorted(makeDataset(val_dir))

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    for index in range(len(train_path)):
        print(index)
        img = Image.open(train_path[index]).convert('RGB')
        replace_path = train_path[index].replace('train', 'base')
        img.save(replace_path)

    for index in range(len(val_path)):
        if val_path[index].endswith('.jpg'):
            number = number + 1
        str_number = str(number)
        zfill_number = str_number.zfill(4)

        img = Image.open(val_path[index]).convert('RGB')
        replace_path = val_path[index].replace('val', 'base')

        replace_phrase = replace_path[-8] + replace_path[-7] + replace_path[-6] + replace_path[-5]

        replace_path = replace_path.replace(replace_phrase, zfill_number)
        replace_path = replace_path.replace('cmp_x', 'cmp_b')

        img.save(replace_path)



if __name__ == '__main__':
    main()