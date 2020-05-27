import os
import sys
import glob

import numpy as np
from scipy.integrate import simps
from matplotlib import pyplot as plt

def LandmarkError(groundtruth, predicted, normalization='centers', showResults=False, verbose=False):
    errors = []
    data_lens = len(groundtruth)

    for i in range(data_lens):

        gtLandmarks = groundtruth[i]
        resLandmarks = predicted[i]

        if normalization == 'centers':
            normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
        elif normalization == 'corners':
            normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
        elif normalization == 'diagonal':
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist = np.sqrt(width ** 2 + height ** 2)

        error = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks)**2,axis=1))) / normDist       
        errors.append(error)
        if verbose:
            print("{0}: {1}".format(i, error))

        if showResults:
            plt.imshow(img[0], cmap=plt.cm.gray)            
            plt.plot(resLandmarks[:, 0], resLandmarks[:, 1], 'o')
            plt.show()

    if verbose:
        print("Image idxs sorted by error")
        print(np.argsort(errors))
    avgError = np.mean(errors)
    print("-- Average error: {0}".format(avgError))

    return errors


def AUCError(errors, failureThreshold, step=0.0001, showCurve=False):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))

    ced =  [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    print("-- AUC @ {0}: {1}".format(failureThreshold, AUC))
    print("-- Failure rate: {0}".format(failureRate))

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()


def LoadDataset(data_dir):
    ptslist = []
    p = os.path.join(data_dir, '*.ptv')
    ptslist.extend(glob.glob(p))

    if 'test' in data_dir:
        p = os.path.join(data_dir, 'imgs_mean.ptv')
        ptslist.remove(p)
        p = os.path.join(data_dir, 'imgs_std.ptv')
        ptslist.remove(p)
        p = os.path.join(data_dir, 'mean_shape.ptv')
        ptslist.remove(p)

    data_set = []
    for ptsfile in ptslist:
        pts = (np.loadtxt(ptsfile,dtype=np.float32,delimiter=',')).astype(np.float32)
        data_set.append(pts)

    return data_set


verbose = False
showResults = False
showCED = False

#'diagonal', 'corners', 'centers'
normalization = 'diagonal'
failureThreshold = 0.08

common_gt_dir = './prep/test/common_set'
challenge_gt_dir = './prep/test/challenge_set'
private_gt_dir = './prep/test/300w_private_set'

common_gt_set = LoadDataset(common_gt_dir)
challenge_gt_set = LoadDataset(challenge_gt_dir)
private_gt_set = LoadDataset(private_gt_dir)

common_res_dir = './prep/predict/common_set'
challenge_res_dir = './prep/predict/challenge_set'
private_res_dir = './prep/predict/300w_private_set'

common_res_set = LoadDataset(common_res_dir)
challenge_res_set = LoadDataset(challenge_res_dir)
private_res_set = LoadDataset(private_res_dir)

print ("Processing common subset of the 300W public test set (test sets of LFPW and HELEN)")
commonErrs = LandmarkError(common_gt_set, common_res_set, normalization, showResults, verbose)
print ("Processing challenging subset of the 300W public test set (IBUG dataset)")
challengingErrs = LandmarkError(challenge_gt_set, challenge_res_set, normalization, showResults, verbose)

fullsetErrs = commonErrs + challengingErrs
print ("Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN)")
print("-- Average error: {0}".format(np.mean(fullsetErrs)))
AUCError(fullsetErrs, failureThreshold, showCurve=showCED)

print ("Processing 300W private test set")
w300Errs = LandmarkError(private_gt_set, private_res_set, normalization, showResults, verbose)
AUCError(w300Errs, failureThreshold, showCurve=showCED)
