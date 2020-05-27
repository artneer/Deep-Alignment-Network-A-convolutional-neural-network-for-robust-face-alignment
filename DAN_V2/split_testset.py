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


challenge_set_dir = "{}/data/test/challenge_set".format(os.getcwd())
_make_safely_folder(challenge_set_dir)
challenge_img_set = _get_filenames(["./db/ibug"], ["*.jpg", "*.png"])
for x in challenge_img_set:
    filepath_without_ext,_ = os.path.splitext(x)
    basefilename = os.path.basename(x)
    filename,_ = os.path.splitext(basefilename)

    img1 = x
    img2 = "{}/{}".format(challenge_set_dir, basefilename)
    pts1 = "{}.pts".format(filepath_without_ext)
    pts2 = "{}/{}.pts".format(challenge_set_dir, filename)
    
    shutil.copy(img1, img2)
    shutil.copy(pts1, pts2)


common_set_dir = "{}/data/test/common_set".format(os.getcwd())
_make_safely_folder(common_set_dir)
common_img_set = _get_filenames(["./db/lfpw/trainset", "./db/helen/trainset"], ["*.jpg", "*.png"])
for x in common_img_set:
    filepath_without_ext,_ = os.path.splitext(x)
    basefilename = os.path.basename(x)
    filename,_ = os.path.splitext(basefilename)

    img1 = x
    img2 = "{}/{}".format(common_set_dir, basefilename)
    pts1 = "{}.pts".format(filepath_without_ext)
    pts2 = "{}/{}.pts".format(common_set_dir, filename)
    
    shutil.copy(img1, img2)
    shutil.copy(pts1, pts2)


private_set_dir = "{}/data/test/300w_private_set".format(os.getcwd())
_make_safely_folder(private_set_dir)
private_img_set = _get_filenames(["./db/300W/01_Indoor", "./db/300W/02_Outdoor"], ["*.jpg", "*.png"])
for x in private_img_set:
    filepath_without_ext,_ = os.path.splitext(x)
    basefilename = os.path.basename(x)
    filename,_ = os.path.splitext(basefilename)

    img1 = x
    img2 = "{}/{}".format(private_set_dir, basefilename)
    pts1 = "{}.pts".format(filepath_without_ext)
    pts2 = "{}/{}.pts".format(private_set_dir, filename)
    
    shutil.copy(img1, img2)
    shutil.copy(pts1, pts2)


print("Done!!")