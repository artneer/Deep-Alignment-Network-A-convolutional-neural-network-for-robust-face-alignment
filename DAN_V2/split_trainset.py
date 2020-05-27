import math
import os 
import sys
import glob
import random
import numpy as np
import cv2
import shutil

def _make_safely_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def _get_filenames(db_dir, listext):

    imagelist = []

    for data_dir in db_dir:

        for ext in listext:
            p = os.path.join(data_dir, ext)
            imagelist.extend(glob.glob(p))

    return imagelist



imagenames = _get_filenames(["./db/afw", "./db/lfpw/trainset", "./db/helen/trainset"], ["*.jpg", "*.png"])

train_set = imagenames
valid_num = 100
valid_set = random.sample(train_set, valid_num)
for x in valid_set:
    train_set.remove(x)


valid_set_dir = "{}/data/valid/".format(os.getcwd())
_make_safely_folder(valid_set_dir)

for x in valid_set:
    filepath_without_ext,_ = os.path.splitext(x)
    basefilename = os.path.basename(x)
    filename,_ = os.path.splitext(basefilename)

    img1 = x
    img2 = "{}/{}".format(valid_set_dir, basefilename)
    pts1 = "{}.pts".format(filepath_without_ext)
    pts2 = "{}/{}.pts".format(valid_set_dir, filename)
    
    shutil.copy(img1, img2)
    shutil.copy(pts1, pts2)


train_set_dir = "{}/data/train/".format(os.getcwd())
_make_safely_folder(train_set_dir)

for x in train_set:
    filepath_without_ext,_ = os.path.splitext(x)
    basefilename = os.path.basename(x)
    filename,_ = os.path.splitext(basefilename)

    img1 = x
    img2 = "{}/{}".format(train_set_dir, basefilename)
    pts1 = "{}.pts".format(filepath_without_ext)
    pts2 = "{}/{}.pts".format(train_set_dir, filename)
    
    shutil.copy(img1, img2)
    shutil.copy(pts1, pts2)


print("Done!!")