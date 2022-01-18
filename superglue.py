import numpy as np
import os
import cv2
import re
import operator
import warnings
from scipy.io import loadmat
import utm
import matplotlib.pyplot as plt
from matplotlib import collections as mc
#import pylab as pl
import math
from os import path
import sys

warnings.filterwarnings("ignore")


# read NPZ file, go through tests to see if this represents a successfully-registered pair of images or not
# no temporal filtering possible here

# this section is taken from superglue.py / mosaic_superglue.py
#num_matches_threshold = 5
glue2imscale = 2.0

def check_registration_failure(inlier_fraction, median_match_confidence):
    if inlier_fraction < 0.5 or median_match_confidence < 0.5:
#        print("IF=", inlier_fraction, " MMC=", median_match_confidence)
        return True
    else:
        return False

def compute_overlap(npzfullfilename, debug=False, num_matches_threshold=5):
    npz = np.load(npzfullfilename)
        
    num_matches = np.sum(npz['matches'] > -1)
    if debug:
        print("num matches", num_matches)
    if num_matches < num_matches_threshold:
        return None

    matched_indices0 = np.where(npz['matches'] > -1)
    matched_kpts0 = glue2imscale*npz['keypoints0'][matched_indices0]
    matched_indices1 = npz['matches'][matched_indices0]
    matched_kpts1 = glue2imscale*npz['keypoints1'][matched_indices1]
    mmc = np.median(npz['match_confidence'][matched_indices0])
    if debug:
        print("MMC", mmc)
    
    H_affine,H_inliers=cv2.estimateAffinePartial2D(matched_kpts0, matched_kpts1, method=cv2.RANSAC,ransacReprojThreshold=20.0)
    if not np.any(H_affine):
        return None
    H = np.append(H_affine, [[0.0, 0.0, 1.0]], axis=0)
    if debug:
        print("H", H)
    
    num_inliers = np.sum(H_inliers)
    inlier_fraction = num_inliers / num_matches
    if debug:
        print("num inliers", num_inliers)
        print("IF", inlier_fraction)
    
    if check_registration_failure(inlier_fraction, mmc):
        return None

    return H
