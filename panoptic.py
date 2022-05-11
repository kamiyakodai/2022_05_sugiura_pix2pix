# -*- coding: utf-8 -*-
"""panoptic用.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sd5b7IJbvOxkPQm1zxSZ7OFTPlwhORjJ

# Install detectron2
"""

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
import pickle

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import argparse


def multiChannelConvert(image_tensor):
    """
    multiChannelConvert
    """

    img = image_tensor.to('cpu').detach().numpy().copy()
    img_n_class = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] not in img_n_class:
                img_n_class.append(img[i][j])

    img_n_class = sorted(img_n_class)
    C, H, W = len(img_n_class), img.shape[0], img.shape[1]
    multi_channel_image = np.zeros((C, H, W), dtype=float)

    for k in range(len(img_n_class)):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_n_class[k] == img[i][j]:
                    multi_channel_image[k][i][j] = 1

    return multi_channel_image


def keepPickle(multi_channel_image, path):
    """#pickle保存"""

    with open(path, mode='wb') as f:
        pickle.dump(multi_channel_image, f)


def main():
    """
    Run a pre-trained detectron2 model
    Then, we create a detectron2 config
    and a detectron2 `DefaultPredictor` to run inference on this image.
    """

    parser = argparse.ArgumentParser(description='panopitc segmentation')
    parser.add_argument('-r', '--path', type=str,
                        default='/mnt/HDD4TB-3/sugiura/pix2pix/imageUCFdataset/',
                        help='path of dataset.')
    parser.add_argument('-p', '--dir', type=str,
                        default='/mnt/HDD4TB-3/sugiura/pix2pix/mycreat_UCFdataset_multichannel_pickle2',
                        help='path of dataset.')
    args = parser.parse_args()

    print(args)

    path = args.path
    assert os.path.isdir(path)

    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    files_dir = sorted(files_dir)

    # print(files_dir)


    for i in range(len(files_dir)):
        fname_path = os.path.join(path, files_dir[i])
        files = os.listdir(fname_path)
        files_file = [f for f in files if os.path.isfile(os.path.join(fname_path, f))]
        print(len(files_file))

        for index in range(len(files_file)):
            mkdir_path = args.dir
            dirname = os.path.join(mkdir_path, files_dir[i])
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            fname = files_dir[i] + '_' + str(index) + '.pickle'
            pickle_path = os.path.join(dirname, fname)

            if os.path.exists(pickle_path):
                continue

            fname = files_dir[i] + '_' + str(index) + '.png'
            file_path = os.path.join(path, files_dir[i], fname)
            print(file_path)
            assert os.path.exists(file_path)


            im = cv2.imread(file_path)
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
            predictor = DefaultPredictor(cfg)
            panoptic_seg, segments_info = predictor(im)["panoptic_seg"]

            multi_channel_image = multiChannelConvert(panoptic_seg)

            keepPickle(multi_channel_image, pickle_path)

        # torch.set_printoptions(edgeitems=1000)
        # print(panoptic_seg)
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        # cv2_imshow(out.get_image()[:, :, ::-1])



    # Inference with a panoptic segmentation model
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    # predictor = DefaultPredictor(cfg)

    # panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    # print(panoptic_seg.size())
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # print(type(v))
    # out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    # cv2_imshow(out.get_image()[:, :, ::-1])



if __name__ == '__main__':
  main()
