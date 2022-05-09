import cv2
import sys
import torch
import os

def get_dataset():
    dir = "UCFdataset"
    UCFdataset_path = []
    UCFdataset_fname = []

    #ディレクトリの確認
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    #root:現在のディレクトリ
    # _:内包するディレクトリ
    #fnames:内包するファイル
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            UCFdataset_path.append(path)
            UCFdataset_fname.append(fname)

    UCFdataset_path = sorted(UCFdataset_path)
    UCFdataset_fname = sorted(UCFdataset_fname)

    return UCFdataset_path, UCFdataset_fname

def video_take_in():
    path, fnames = get_dataset()
    for index in range(len(path)):
        print(index)
        video_path = path[index]
        fname = fnames[index]
        fname = fname[0:len(fname)-4]
        # 動画ファイル読込
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            # 正常に読み込めたのかチェックする
            # 読み込めたらTrue、失敗ならFalse
            print("動画の読み込み失敗")
            sys.exit()

        # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # fps = cap.get(cv2.CAP_PROP_FPS)

        # print("width:{}, height:{}, count:{}, fps:{}".format(width,height,count,fps))

        dirname = 'image_dataset'
        dirname = os.path.join(dirname, str(fname))
        if not os.path.exists(dirname):
            os.makedirs(dirname)


        n= 0
        while True:
            # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
            is_image,frame_img = cap.read()
            if is_image:
                # 画像を保存
                filename = str(fname) + "_" +str(n) + ".png"
                cv2.imwrite(os.path.join(dirname, filename), frame_img)

            else:
                # フレーム画像が読込なかったら終了
                break
            n += 1

video_take_in()
