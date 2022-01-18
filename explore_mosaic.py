import matplotlib
matplotlib.use("agg")

import numpy as np
from scipy import ndimage
import os
import cv2
import re
import operator
import warnings
from scipy.io import loadmat
import utm
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.path import Path as plotpath
#import pylab as pl
import math    
from os import path
import sys
import csv
import superglue
from datetime import datetime, timedelta
from operator import itemgetter
import time
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

import faulthandler
faulthandler.enable()
#import pdb
#pdb.set_trace()
#sys.settrace()
warnings.filterwarnings("ignore")

far_next_set = set()
leg_end_set = set()

max_leg_dist = 30.0

leg_dx_list = []
leg_dy_list = []
leg_start_idx_list = []
leg_end_idx_list = []
leg_start_sonar_idx_list = []
leg_end_sonar_idx_list = []
leg_m_list = []
leg_c_list = []
leg_num_total_list = []
leg_num_outliers_list = []
leg_context_index = -1
min_leg_images = 2


mosaic_utm_x_left = -1
mosaic_utm_x_right = -1
mosaic_utm_y_top = -1
mosaic_utm_y_bottom = -1

sonar_context_index = -1

sonar_alignment_factor_list = []  # 0.0
sonar_alignment_factor = 0.0

do_sonar = True
draw_frame_info = False

do_heading_rotation = True  # False

impathprefix = "/data/scallops/images/"

mission_name = "20170817_IM"

if len(sys.argv) >= 3:
    mission_name = sys.argv[2]
    print("mission", mission_name)

# if 20170817 or anything else, as far as we know
sonarrectwidth = 40   
sonarrectlength = 40

if mission_name == "20140711_IM":
    neighbor_span = "xxx_yyy"
elif mission_name == "20150711_IM_2":
    neighbor_span = "xxx_yyy"
elif mission_name == "20170817_IM":
    neighbor_span = "1353_4408"
elif mission_name == "20170824_VIMS":
    neighbor_span = "62730_63416"
    sonarrectwidth = 80   
    sonarrectlength = 80
else:
    print("unknown mission")
    sys.exit()

monthdict = {}
monthdict[1] = "JAN"
monthdict[2] = "FEB"
monthdict[3] = "MAR"
monthdict[4] = "APR"
monthdict[5] = "MAY"
monthdict[6] = "JUN"
monthdict[7] = "JUL"
monthdict[8] = "AUG"
monthdict[9] = "SEP"
monthdict[10] = "OCT"
monthdict[11] = "NOV"
monthdict[12] = "DEC"


metadatafilename = "/data/scallops/metadata/" + mission_name + "_camerametadata.mat"
neighbors_filename = mission_name + "_" + neighbor_span + "_neighbors.csv"
sonar_superdir = "/data/scallops/sonar/" + mission_name + "/"
new_sonar_superdir = '/home/jiayi/sonar/retinanet/image_predictions/'
files = os.listdir(sonar_superdir)
#print('files?',len(files))

sonar_prefix = mission_name[6:8] + monthdict[int(mission_name[4:6])]
sonar2cam_filename = mission_name + "_sonar2cam.csv"

sonar_overlap_row = []
sonar_discontinuity_flag = []

numsonars = 0
sonarlist = []
dirlisting = sorted(os.listdir(sonar_superdir))
for filename in dirlisting:
    if filename.endswith(".mat"):
        sonarlist.append(filename)
        numsonars += 1
        #print(filename)
if numsonars == 0:
    print("no sonar files in specified directory")
    sys.exit()
#print(numsonars)
#sys.exit()

current_sonar_exists = True
sonar_fixed_nadir_radius = 300

#metadatafilename = "/data/scallops/metadata/20170824_VIMS_camerametadata.mat"
#neighbors_filename = "20170824_VIMS_62730_63416_neighbors.csv"

#metadatafilename = "/data/scallops/metadata/20170817_IM_camerametadata.mat"
#neighbors_filename = "20170817_IM_1353_4408_neighbors.csv"
#sonar_superdir = "/data/scallops/sonar/20170817_IM/"
#sonar_prefix = "17AUG"
#numsonars = 567
#sonar2cam_filename = "20170817_IM_sonar2cam.csv"

#metadatafilename = "/data/scallops/metadata/20170824_VIMS_camerametadata.mat"
#neighbors_filename = "20170824_VIMS_62730_63416_neighbors.csv"
#sonar_superdir = "/data/scallops/sonar/20170824_VIMS/"
#sonar_prefix = "24AUG"
#numsonars = 600   # it would be nice if we could figure this out ourselves
#sonar2cam_filename = "20170824_VIMS_sonar2cam.csv"

imwidth = 1280
imheight = 960
imcorners = np.transpose([[0, 0, 1], [imwidth-1, 0, 1], [imwidth-1, imheight-1, 1], [0, imheight-1, 1]])
transposed_imcorners = np.transpose(imcorners)

# this could be computed from the PLL rectangles in a more principle way
IMAGE_WIDTH_IN_METERS = 1.9

mask_im = np.zeros((imheight, imwidth, 1), np.uint8)
mask_im[:,:] = (255)

imcenter = np.transpose([imwidth/2, imheight/2, 1])

# before scaling
mosaic_xtrans = imwidth/4
mosaic_ytrans = imheight/2

mosaic_window_width = int(3*imwidth/4)
#mosaic_window_height = int(imheight)
mosaic_window_height = int(5*imheight/4)

# starting with lower left to match apparent PLL order
    
pll_imcorners = np.transpose([[0, imheight-1, 1], [0, 0, 1], [imwidth-1, 0, 1], [imwidth-1, imheight-1, 1]])
transposed_pll_imcorners = np.transpose(pll_imcorners)

sonar_datetime = []
sonar_filename = []

draw_camera_over_sonar = True

sonar_im_height = 1000
sonar_im_width = 1024

current_sonarrect_idx = 0

image_in_sonarrect_idx_list = []
image_in_sonarrect_waterfall_im_list = []
image_in_sonarrect_waterfall_nadir_mask_im_list = []

#plt_fig, plt_ax = plt.subplots()
#plt.ion()

#print(imcorners)
#print(pll_imcorners)
#print(transposed_pll_imcorners)
#sys.exit()

#-------------------------------------------------------------------------------------
# functions
#-------------------------------------------------------------------------------------      
# RetinaNet Annotations                                                                         
def load_retinanet():
    global annotation_list
    csv_path = '/home/jiayi/sonar/retinanet/retinanet_predictions.csv'
    with open(csv_path, 'r') as csvfile:
        annotation_list = csvfile.read().splitlines()                                    
#-------------------------------------------------------------------------------------
heading_distthresh = 5.0

def compute_image_positions_and_headings():

    # positions
    
    for t in range(0, num_imageframes):

        image_altitude[t] = camera['altitude'][t]
        
        lat = camera['latitude'][t]
        lon = camera['longitude'][t]

        if math.isnan(camera['PLL'][0][0][t]) or math.isnan(camera['PLL'][0][1][t]) or \
           math.isnan(camera['PLL'][1][0][t]) or math.isnan(camera['PLL'][1][1][t]) or \
           math.isnan(camera['PLL'][2][0][t]) or math.isnan(camera['PLL'][2][1][t]) or \
           math.isnan(camera['PLL'][3][0][t]) or math.isnan(camera['PLL'][3][1][t]):
            imagex[t] = math.nan
            imagey[t] = math.nan
#            print("NaN for", t, "lat/lon are ", lat, lon)
            
        else:
            p0 = utm.from_latlon(camera['PLL'][0][0][t], camera['PLL'][0][1][t])
            p1 = utm.from_latlon(camera['PLL'][1][0][t], camera['PLL'][1][1][t])
            p2 = utm.from_latlon(camera['PLL'][2][0][t], camera['PLL'][2][1][t])
            p3 = utm.from_latlon(camera['PLL'][3][0][t], camera['PLL'][3][1][t])

            imagex[t] = 0.25 * (p0[0] + p1[0] + p2[0] + p3[0])
            imagey[t] = 0.25 * (p0[1] + p1[1] + p2[1] + p3[1])

    # headings

    for t in range(0, num_imageframes):

        # not a number

        if math.isnan(imagex[t]) or math.isnan(imagey[t]):
            imagedx[t] = math.nan
            imagedy[t] = math.nan

        # *is* number -- approximate heading with position difference
    
        else:
            next=t+1
            previous=t-1

            # beginning of sequence
        
            if t == 0:
                previous = t
                print(t, "START of sequence")

            # end of sequence

            elif t == num_imageframes-1:
                next = t
                print(t, "END of sequence")

            # middle of sequence, but adjacent data point(s) might be too far away for valid heading approximation

            else: 
                prevdx = imagex[t] - imagex[previous]
                prevdy = imagey[t] - imagey[previous]
                prevlen = math.sqrt(prevdx*prevdx + prevdy*prevdy)
                if prevlen > heading_distthresh:
                    previous = t
                    #print(t, "FAR PREVIOUS", prevlen)
   
                nextdx = imagex[next] - imagex[t]
                nextdy = imagey[next] - imagey[t]
                nextlen = math.sqrt(nextdx*nextdx + nextdy*nextdy)
                if nextlen > heading_distthresh:
                    if image_altitude[t] < 25 and image_altitude[next] < 25:
                        print(t, "FAR NEXT", nextlen, image_altitude[t])
                        far_next_set.add(t)
                    next = t
                    image_farnext[t] = 1
                
            dx = imagex[next] - imagex[previous]
            dy = imagey[next] - imagey[previous]
            mylen = math.sqrt(dx*dx + dy*dy)

            # if *too close*, no valid heading estimate
            if abs(mylen) < 0.001:
                imagedx[t] = math.nan
                imagedy[t] = math.nan
#                print(t, "TOO CLOSE")
                
            # valid heading estimate
            else:
                imagedx[t]=dx/mylen
                imagedy[t]=dy/mylen
    
#-------------------------------------------------------------------------------------

def get_frame_timeinfo(idx):
    frame_imsuperdir, frame_imsubdir, frame_imname, frame_file_found_flag = get_image_filename(camera['filename'][idx][0][0])
    #if frame_file_found_flag == False:
    #    print("file not found!", idx, camera['filename'][idx][0][0])
    #    sys.exit()
        
    frame_datetime = datetime.strptime(frame_imsubdir[:-1], '%Y%m%d%H%M%S')
    frame_capturetime = camera['capturetime'][idx][0]
    
    return frame_capturetime, frame_datetime

#-------------------------------------------------------------------------------------

# assuming camera meta data has already been loaded

# this is how we get image index from sonar index
cam_minindex = []
cam_mindiff = []

sonarx = np.zeros(numsonars)
sonary = np.zeros(numsonars)
sonardx = np.zeros(numsonars)
sonardy = np.zeros(numsonars)
sonar_R_rect2utm_list = []
sonar_R_utm2rect_list = []

# sonar positions projected onto their respective leg lines
sonarx_leg_proj = np.zeros(numsonars)
sonary_leg_proj = np.zeros(numsonars)

# let's just approximate each MST file as an oriented rectangle: 20 m wide and 20 m square
# we should be able to derive the corners of this rectangle from each sonar (x, y, dx, dy) like so:
#
#         
#         
#  3.___________.0    * = (0, 0) after transformation
#   |           |     x = image point
#   |           |
#   |  x        *->
#   |           |
#   |           |   x
#  2.___________.1
#
# take image center coordinates, subtract sonarx and sonary, then apply sonar heading anti-rotation
# before testing if transformed point is inside rectangle drawn above.
# this is good for test for whether the image center is inside it, but we need to apply opposite of
# same transform to plot rect corners

# 0 is "front" left, 1 is "front" right
sonarcorners = np.transpose([[0, sonarrectwidth/2, 1], [0, -sonarrectwidth/2, 1], [-sonarrectlength, -sonarrectwidth/2, 1], [-sonarrectlength, sonarrectwidth/2, 1]])
#print(sonarcorners)

waterfallcorners = np.array([[0, sonar_im_width - 1, sonar_im_width - 1,  0], \
                             [0, 0,                  sonar_im_height - 1, sonar_im_height - 1], \
                             [1, 1, 1, 1]])
(H_waterfall_to_rect, _) = cv2.findHomography(np.transpose(waterfallcorners)[:,0:2], np.transpose(sonarcorners)[:,0:2])

#sonarrectx = [0, 0, -sonarrectlength, -sonarrectlength]      
#sonarrecty = [-sonarrectwidth/2, sonarrectwidth/2, sonarrectwidth/2, -sonarrectwidth/2]

#1st corner repeated so that it's a closed polygon
sonar_rectpath = plotpath([[sonarcorners[0][0], sonarcorners[1][0]], [sonarcorners[0][1], sonarcorners[1][1]], [sonarcorners[0][2], sonarcorners[1][2]], [sonarcorners[0][3], sonarcorners[1][3]], [sonarcorners[0][0], sonarcorners[1][0]]])

#-------------------------------------------------------------------------------------

# should be vertically oriented, nadir on the left for both (not expecting left im to have been LR flipped yet)

min_sonar_nadir_radius = 50
#sonar_nadir_max_value = 60
sonar_nadir_max_value = 40
sonar_nadir_window_size = 25

def detect_sonar_nadir_edges(sonar_left_im, sonar_right_im, do_display=False):

#    sonar_left_im = raw_sonar_left_im
#    sonar_right_im = raw_sonar_right_im
    # decision has been made not to filter the waterfall image itself
#    sonar_left_im = ndimage.median_filter(raw_sonar_left_im, 3)
#    sonar_right_im = ndimage.median_filter(raw_sonar_right_im, 3)

    raw_sonar_left_nadir_edges = np.full(sonar_left_im.shape[0], sonar_left_im.shape[1]) 
    raw_sonar_right_nadir_edges = np.full(sonar_left_im.shape[0], sonar_left_im.shape[1]) 

    # look for first return over threshold
    
    for row in range(sonar_left_im.shape[0]):
        for col in range (min_sonar_nadir_radius, sonar_left_im.shape[1]):
            if sonar_left_im[row, col] > sonar_nadir_max_value and raw_sonar_left_nadir_edges[row] == sonar_left_im.shape[1]:
                raw_sonar_left_nadir_edges[row] = col
            if sonar_right_im[row, col] > sonar_nadir_max_value and raw_sonar_right_nadir_edges[row] == sonar_left_im.shape[1]:
                raw_sonar_right_nadir_edges[row] = col

    # smooth edge with running average (median)
    
    smooth_sonar_left_nadir_edges = ndimage.median_filter(raw_sonar_left_nadir_edges, sonar_nadir_window_size)
    smooth_sonar_right_nadir_edges = ndimage.median_filter(raw_sonar_right_nadir_edges, sonar_nadir_window_size)

    # display results -- debug only

    if do_display:
        color_left_im = cv2.cvtColor(sonar_left_im, cv2.COLOR_GRAY2BGR)
        cv2.line(color_left_im, \
                 (min_sonar_nadir_radius, 0), (min_sonar_nadir_radius, sonar_left_im.shape[0] - 1), 
                 (0, 255, 255), 1)
        
        for row in range(sonar_left_im.shape[0] - 1):
            cv2.line(color_left_im, \
                     (smooth_sonar_left_nadir_edges[row], row), (smooth_sonar_left_nadir_edges[row + 1], row + 1), 
                     (0, 0, 255), 1)
            
            color_right_im = cv2.cvtColor(sonar_right_im, cv2.COLOR_GRAY2BGR)
            cv2.line(color_right_im, \
                     (min_sonar_nadir_radius, 0), (min_sonar_nadir_radius, sonar_left_im.shape[0] - 1), 
                     (0, 255, 255), 1)

            for row in range(sonar_left_im.shape[0] - 1):
                cv2.line(color_right_im, \
                         (smooth_sonar_right_nadir_edges[row], row), (smooth_sonar_right_nadir_edges[row + 1], row + 1), 
                         (0, 0, 255), 1)

        color_left_im = np.fliplr(color_left_im)
        color_im = np.hstack((color_left_im, color_right_im))

        cv2.imshow("nadir", color_im)
#    key = cv2.waitKey(5)

#    if key==ord('q'):
#        sys.exit()    

    return smooth_sonar_left_nadir_edges, smooth_sonar_right_nadir_edges
    
#-------------------------------------------------------------------------------------

def make_nadir_mask(sonar_im, sonar_left_nadir_edges, sonar_right_nadir_edges):

    sonar_nadir_mask_im = np.zeros(sonar_im.shape, dtype=np.uint8)

    centerline = sonar_im.shape[1]/2
    for row in range(sonar_im.shape[0]):
        nadir_left = int(centerline - sonar_left_nadir_edges[row])
        nadir_right = int(centerline + sonar_right_nadir_edges[row])
        sonar_nadir_mask_im[row, nadir_left:nadir_right] = 255
    
    return sonar_nadir_mask_im

#-------------------------------------------------------------------------------------

# works on left and right as long as there has been no left right flip

sonar_dont_trust_radius = 50

def slant_range_correction(sonar_left_im, sonar_right_im, sonar_left_nadir_edges, sonar_right_nadir_edges):

    # left
    
    corrected_sonar_left_im = np.zeros(sonar_left_im.shape, dtype=np.uint8)  # should it be NaNs?
#    left_beam_thetas = []
#    left_beam_energies = []
    left_map_x = np.zeros((sonar_left_im.shape[0], sonar_left_im.shape[1]), np.float32)
    left_map_y = np.zeros((sonar_left_im.shape[0], sonar_left_im.shape[1]), np.float32)
#    print(sonar_left_im.shape, left_map_x.shape)
    
#    left_map_x = np.zeros(sonar_left_im.shape)  
#    left_map_y = np.zeros(sonar_left_im.shape)
    #print(type(sonar_left_im))
#    print(left_map_x.type())
#    print(left_map_y.type())

    for row in range(sonar_left_im.shape[0]):
        t_height = float(sonar_left_nadir_edges[row])
        t_height_squared = t_height*t_height
#        for col in range (sonar_left_nadir_edges[row], sonar_left_im.shape[1]):
#            t_slant_squared = float(col*col)
#            t_ground = int(round(math.sqrt(t_slant_squared - t_height_squared)))
#            # math.sqrt(dst_x_squared + t_height_squared) = src_x
#            theta = math.degrees(math.atan2(t_ground, t_height))
#            left_beam_thetas.append(theta)
#            left_beam_energies.append(sonar_left_im[row, col])
##           corrected_sonar_left_im[row, t_ground] = sonar_left_im[row, col]
#            left_map_x[row, col] = float(t_ground)
#            left_map_y[row, col] = float(row)


        for col in range (sonar_left_im.shape[1]):
            left_map_x[row, col] = math.sqrt(float(col*col+t_height_squared))
            left_map_y[row, col] = float(row)

    corrected_sonar_left_im = cv2.remap(sonar_left_im, left_map_x, left_map_y, cv2.INTER_LINEAR)
    corrected_sonar_left_im[:,:sonar_dont_trust_radius] = 0
            
    # right

    corrected_sonar_right_im = np.zeros(sonar_right_im.shape, dtype=np.uint8)  # should it be NaNs?
    right_map_x = np.zeros((sonar_right_im.shape[0], sonar_right_im.shape[1]), np.float32)
    right_map_y = np.zeros((sonar_right_im.shape[0], sonar_right_im.shape[1]), np.float32)

    for row in range(sonar_right_im.shape[0]):
        t_height = float(sonar_right_nadir_edges[row])
        t_height_squared = t_height*t_height

        for col in range (sonar_right_im.shape[1]):
            right_map_x[row, col] = math.sqrt(float(col*col+t_height_squared))
            right_map_y[row, col] = float(row)


    corrected_sonar_right_im = cv2.remap(sonar_right_im, right_map_x, right_map_y, cv2.INTER_LINEAR)
    corrected_sonar_right_im[:,:sonar_dont_trust_radius] = 0

    # display


    corrected_sonar_left_im = np.fliplr(corrected_sonar_left_im)
#    cv2.imshow("SR corrected range", corrected_sonar_left_im)
    corrected_sonar_im = np.hstack((corrected_sonar_left_im, corrected_sonar_right_im))
#    sonar_im = np.hstack((np.fliplr(sonar_left_im), sonar_right_im))

#    plt.scatter(left_beam_thetas, left_beam_energies,s=1)
#    plt.show()
    
#    cv2.imshow("SR raw image", sonar_im)
    cv2.imshow("SR corrected range", corrected_sonar_im)
    key = cv2.waitKey(0)

    if key==ord('q'):
        sys.exit()    

    
#-------------------------------------------------------------------------------------

def generate_pair_list():
    index_set = {36, 37, 38, 64, 65, 66, 92, 93, 94, 126, 127, 128, 154, 155, 156, 182, 183, 184, 210, 211}

    # iterate through the set and write a pair for every possible index/index and left/right combination except
    # left and right of the same index

    f = open("/tmp/20170817_bigger_nadir_pairs.txt", "w")
    
    for idx1 in index_set:
        for idx2 in index_set:
            if idx1 != idx2:
                idx1_str = str(idx1).zfill(3)
                idx2_str = str(idx2).zfill(3)
                # LL
                idx1_name = "LEFT_300_17AUG" + idx1_str + "_MST_data.png"
                idx2_name = "LEFT_300_17AUG" + idx2_str + "_MST_data.png"
                f.write(idx1_name + " " + idx2_name + "\n")
                # LR
                idx2_name = "RIGHT_300_17AUG" + idx2_str + "_MST_data.png"
                f.write(idx1_name + " " + idx2_name + "\n")
                # RR
                idx1_name = "RIGHT_300_17AUG" + idx1_str + "_MST_data.png"
                f.write(idx1_name + " " + idx2_name + "\n")
                # RL
                idx2_name = "LEFT_300_17AUG" + idx2_str + "_MST_data.png"
                f.write(idx1_name + " " + idx2_name + "\n")
    f.close()

    print("finished generate_pair_list()") 
    sys.exit()
    
#-------------------------------------------------------------------------------------

def read_cache_sonar_images():

    for sonar_idx in range(numsonars):

        sonar_idx_str = str(sonar_idx).zfill(3)
        imagename = sonar_prefix + sonar_idx_str + "_MST_data.png"
        in_filename = sonar_superdir + "images/" + imagename

        sonar_waterfall_im = cv2.imread(in_filename)

        out_left_filename = "/tmp/LEFT_" + str(sonar_fixed_nadir_radius) + "_" + sonar_prefix + sonar_idx_str + "_MST_data.png"
        out_right_filename = "/tmp/RIGHT_" + str(sonar_fixed_nadir_radius) + "_" + sonar_prefix + sonar_idx_str + "_MST_data.png"
        print(out_left_filename, out_right_filename)

        left_sonar_waterfall_im = sonar_waterfall_im[:,0:sonar_fixed_nadir_radius,:]
        right_sonar_waterfall_im = sonar_waterfall_im[:, sonar_im_width-sonar_fixed_nadir_radius:,:]
        horiz_left_sonar_waterfall_im = cv2.rotate(left_sonar_waterfall_im, cv2.ROTATE_90_CLOCKWISE)
        horiz_right_sonar_waterfall_im = cv2.rotate(right_sonar_waterfall_im, cv2.ROTATE_90_CLOCKWISE)

#        sonar_waterfall_im[:,sonar_fixed_nadir_radius:sonar_im_width-sonar_fixed_nadir_radius,:] = 0
#        sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_mat(sonar_idx)
        cv2.imshow("left", horiz_left_sonar_waterfall_im)
        cv2.imshow("right", horiz_right_sonar_waterfall_im)
        cv2.imwrite(out_left_filename, horiz_left_sonar_waterfall_im)
        cv2.imwrite(out_right_filename, horiz_right_sonar_waterfall_im)
#        cv2.imshow("nadir", sonar_waterfall_nadir_mask_im)
#        cv2.imwrite(out_filename, sonar_waterfall_im)
        key = cv2.waitKey(5)
        if key==ord('q'):
            break
        
    print("finished read_cache()") 
    sys.exit()

    #-------------------------------------------------------------------------------------

def write_cache_sonar_images():

    for sonar_idx in range(numsonars):

        sonar_idx_str = str(sonar_idx).zfill(3)
        sonarname = sonar_prefix + sonar_idx_str + "_MST_data.mat"
        imagename = sonar_prefix + sonar_idx_str + "_MST_data.png"
        in_filename = sonar_superdir + sonarname
        out_filename = sonar_superdir + "images/" + imagename
        print(in_filename, out_filename)

        sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_mat(sonar_idx)
        cv2.imshow("image", sonar_waterfall_im)
#        cv2.imshow("nadir", sonar_waterfall_nadir_mask_im)
        cv2.imwrite(out_filename, sonar_waterfall_im)
        key = cv2.waitKey(5)
        if key==ord('q'):
            break
    print("finished write_cache()") 
    sys.exit()

#-------------------------------------------------------------------------------------

# (x0, y0) is some point
# m, c defines the line in slope (m)-intercept (c) form

# from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation

# if the line were defined as ax + by + c = 0, then the closest point on the line would be:
#       b(bx0 - ay0) - ac
#  x = -------------------
#         a^2 + b^2
#
#       a(ay0 - bx0) - bc      
#  y = -------------------
#         a^2 + b^2

def closest_point_on_line(m, c, x0, y0):
    if math.isnan(m):
        return math.nan, math.nan
    ratio = m.as_integer_ratio()
    n = ratio[0]
    d = ratio[1]
    A = float(-n)
    B = float(d)
    C = float(-d * c)

    denom = A*A + B*B
    x = (B * (B * x0 - A * y0) - A * C) / denom
    y = (A * (A * y0 - B * x0) - B * C) / denom

    xdiff = x - x0
    ydiff = y - y0
    dist = math.sqrt(xdiff*xdiff + ydiff*ydiff)

    return x, y, dist

#-------------------------------------------------------------------------------------

def compute_sonar_idx_to_cam_idx(sonar_idx):
    
    firstframe_capturetime, firstframe_datetime = get_frame_timeinfo(0)

    sonar_idx_str = str(sonar_idx).zfill(3)
    sonarname = sonar_prefix + sonar_idx_str + "_MST_data.mat"
    filename = sonar_superdir + sonarname
    M = loadmat(filename)

    year = M["data"][0][0]['timeCorrelation']['tm_year'][0][0][0][0]   #offset so add base? e.g. 2017 = 117  
    month = M["data"][0][0]['timeCorrelation']['tm_mon'][0][0][0][0]   #0-11
    mday = M["data"][0][0]['timeCorrelation']['tm_mday'][0][0][0][0]   #1-31
    hour = M["data"][0][0]['timeCorrelation']['tm_hour'][0][0][0][0]   
    min = M["data"][0][0]['timeCorrelation']['tm_min'][0][0][0][0]  
    sec = M["data"][0][0]['timeCorrelation']['tm_sec'][0][0][0][0]
    D = datetime(1900+year, 1+month, mday, hour, min, sec)
    
    image_mindiff = 100000000
    for image_idx in range(num_imageframes):
        idxframe_capturetime, idxframe_datetime = get_frame_timeinfo(image_idx)
        idxframe_capturetime_offset = idxframe_capturetime - firstframe_capturetime
        idxframe_datetime = firstframe_datetime + timedelta(seconds = idxframe_capturetime_offset)
        #    print(idxframe_capturetime)
        #            print(image_idx, idxframe_datetime)
        #    print(idxframe_capturetime_offset)
        diff = (D - idxframe_datetime).total_seconds()
        absdiff = abs(diff)
        if (sonar_idx == 8 and image_idx > 450 and image_idx < 470) or \
           (sonar_idx == 5 and image_idx > 290 and image_idx < 310) or \
           (sonar_idx == 2 and image_idx > 160 and image_idx < 180):
            print(image_idx, diff, absdiff, image_minindex, image_mindiff)
        if absdiff < image_mindiff:
            image_mindiff = absdiff
            image_minindex = image_idx
            
    print(sonar_idx, image_minindex, image_mindiff, D.timestamp())
            
#-------------------------------------------------------------------------------------

# populate the sonar equivalent of leg_start_idx_list and leg_end_idx_list

def compute_sonar_leg_endpts():

    leg_start_sonar_idx_list.append(0)
    for leg_idx in range(len(leg_start_idx_list)):
        for sonar_idx in range(numsonars):
            image_idx = cam_minindex[sonar_idx]
#            print(sonar_idx, image_idx)
#            if leg_start_idx_list[leg_idx] == image_idx:
#                leg_start_sonar_idx_list.append(sonar_idx)
            if leg_end_idx_list[leg_idx] == image_idx:
                leg_end_sonar_idx_list.append(sonar_idx)
                if sonar_idx != numsonars - 1:
                    leg_start_sonar_idx_list.append(sonar_idx + 1)
                break

#    print(leg_start_idx_list)
#    print(leg_start_sonar_idx_list)
#    print(len(leg_start_idx_list), len(leg_start_sonar_idx_list))

#    print(leg_end_idx_list)
#    print(leg_end_sonar_idx_list)
#    print(len(leg_end_idx_list), len(leg_end_sonar_idx_list))
#    sys.exit()

#    print(sonar_overlap_row)
#    print(len(sonar_overlap_row))
#    sys.exit()

#-------------------------------------------------------------------------------------

# for a pair of neighboring sonars, figure out alignment adjustment
# base doesn't move, neighbor should be either base + 1 or base - 1

def compute_sonar_alignment_factor(base_sonar_idx, neighbor_sonar_idx):

    xdiff = sonarx_leg_proj[base_sonar_idx] - sonarx_leg_proj[neighbor_sonar_idx]
    ydiff = sonary_leg_proj[base_sonar_idx] - sonary_leg_proj[neighbor_sonar_idx]
    dist = math.sqrt(xdiff*xdiff + ydiff*ydiff)

    # sonar_overlap_row[i] is the overlap of sonar image i+1 with image i

    # neighbor is prev
    if base_sonar_idx == neighbor_sonar_idx + 1:
        ytrans_height_frac = float(sonar_im_height - sonar_overlap_row[base_sonar_idx-1]) / float(sonar_im_height)
        ytrans_meters = ytrans_height_frac * float(sonarrectlength)
        #print("prev ", ytrans_meters, dist, dist - ytrans_meters)
        return (dist - ytrans_meters)
    # neighbor is next
    elif base_sonar_idx == neighbor_sonar_idx - 1:
        ytrans_height_frac = float(sonar_im_height - sonar_overlap_row[base_sonar_idx]) / float(sonar_im_height)
        ytrans_meters = ytrans_height_frac * float(sonarrectlength)
        #print("next ", ytrans_meters, dist, ytrans_meters - dist)
        return (ytrans_meters - dist)
    # error
    else:
        print("compute_sonar_alignment_factor(): base and neighbor indices do not differ by 1")
        sys.exit()

    return 0.0
    
#-------------------------------------------------------------------------------------
    
def preprocess_sonar():

    print("loading sonar...")

    # get overlap info

    sonar_overlap_filename = mission_name + "_sonar_overlaps.csv"
    with open(sonar_overlap_filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            sonar_overlap_row.append(int(row[1]))

    print(len(sonar_overlap_row))
    #exit(1)
          
    # detect discontinuities

    sonar_mode_overlap = 49
    sonar_overlap_not_found = -100

    sonar_discontinuity_flag.append(False)
    for idx in range(1, len(sonar_overlap_row)):
#        print(idx, sonar_overlap_row[idx])
        if sonar_overlap_row[idx] == sonar_overlap_not_found or \
           (sonar_overlap_row[idx] != sonar_mode_overlap and sonar_overlap_row[idx-1] > sonar_mode_overlap): 
            sonar_discontinuity_flag.append(True)
#            print("here is one!")
        else:
            sonar_discontinuity_flag.append(False)
    
    # did we already do this?
    
    if path.exists(sonar2cam_filename):

        print('path exists for this')
        print(sonar2cam_filename)
        
        with open(sonar2cam_filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            for row in spamreader:
                cam_minindex.append(int(row[1]))
                cam_mindiff.append(float(row[2]))

        # this is the first moment that we have image positions and leg starts/ends
        
        current_sonar_idx = 0
        while True:
            # find next discontinuity
            while current_sonar_idx < numsonars - 1 and sonar_discontinuity_flag[current_sonar_idx] == False:
                current_sonar_idx += 1
            if current_sonar_idx >= numsonars - 1:
                break
            # manual adjustment, see discussion in journal on July 7, 2020
#            if mission_name == "20170817_IM" and cam_minindex[current_sonar_idx] == 460:
#                leg_end_idx_list.append(461)
#                leg_end_set.add(461)
#            else:
            leg_end_idx_list.append(cam_minindex[current_sonar_idx])
            leg_end_set.add(cam_minindex[current_sonar_idx])
            current_sonar_idx += 1
        leg_end_idx_list.append(num_imageframes - 1)
        #print(leg_end_idx_list)

        leg_start_idx = 0
        for idx in range(len(leg_end_idx_list)):
            leg_start_idx_list.append(leg_start_idx)
            #print(leg_start_idx, leg_end_idx_list[idx])
            xvec = np.zeros(1 + leg_end_idx_list[idx] - leg_start_idx)
            yvec = np.zeros(1 + leg_end_idx_list[idx] - leg_start_idx)
            little_j = 0
            for big_j in range(leg_start_idx, leg_end_idx_list[idx] + 1):
                xvec[little_j] = imagex[big_j]
                yvec[little_j] = imagey[big_j]
                little_j += 1
            #print(xvec)
            ransac = linear_model.RANSACRegressor(min_samples=25)

            xygood = np.argwhere(np.logical_and(np.isfinite(xvec), np.isfinite(yvec)))
            xvec = xvec[xygood]
            yvec = yvec[xygood]

            # bail if xvec/yvec has less than some number
            if xvec.shape[0] < min_leg_images:
                total_count = 0
                m = math.nan
                c = math.nan
                outlier_count = math.nan
            else:

                total_count = xvec.shape[0]
                ransac.fit(xvec.reshape(-1,1), yvec.reshape(-1,1))

                m, c = float(ransac.estimator_.coef_), float(ransac.estimator_.intercept_)
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)
                outlier_count = np.count_nonzero(outlier_mask)
                #print("INITIAL m={:.3f}, c={:.3f}, n={:d}, o={:d}".format(m, c, xvec.shape[0], outlier_count))
            
                if abs(m) > 1.0:
                    ransac.fit(yvec.reshape(-1,1), xvec.reshape(-1,1))
                    inv_m, inv_c = float(ransac.estimator_.coef_), float(ransac.estimator_.intercept_)
                    m = 1.0 / inv_m
                    c = -inv_c / inv_m
                    inlier_mask = ransac.inlier_mask_
                    outlier_mask = np.logical_not(inlier_mask)
                    outlier_count = np.count_nonzero(outlier_mask)
                    #print("REVERSED m={:.3f}, c={:.3f}, n={:d}, o={:d}".format(m, c, xvec.shape[0], outlier_count))


            leg_m_list.append(m)
            leg_c_list.append(c)
            leg_num_total_list.append(total_count)
            leg_num_outliers_list.append(outlier_count)
            
            leg_start_idx = leg_end_idx_list[idx] + 1

        # compute dx, dy for each leg

        print(len(leg_m_list))
        print(len(leg_c_list))
        print('xxx')
        
        for leg_idx in range(len(leg_start_idx_list)):
            print("leg", leg_idx, leg_start_idx_list[leg_idx])
            xstart = imagex[leg_start_idx_list[leg_idx]]
            ystart = imagey[leg_start_idx_list[leg_idx]]
            if math.isnan(xstart) or math.isnan(ystart):
                continue
            
            print(xstart, ystart)
            xstart_proj, ystart_proj, _ = closest_point_on_line(leg_m_list[leg_idx], leg_c_list[leg_idx], xstart, ystart)

            xend = imagex[leg_end_idx_list[leg_idx]]
            yend = imagey[leg_end_idx_list[leg_idx]]

            if math.isnan(xend) or math.isnan(yend):
                continue
            
            print(xend, yend)
            xend_proj, yend_proj, _ = closest_point_on_line(leg_m_list[leg_idx], leg_c_list[leg_idx], xend, yend)

            dx = xend_proj - xstart_proj
            dy = yend_proj - ystart_proj
            leglength = math.sqrt(dx*dx + dy*dy)
            leg_dx_list.append(dx / leglength)
            leg_dy_list.append(dy / leglength)

        # critical for cross-leg connections!
        
        compute_all_prev_and_next_leg_neighbors()

        # first pass to populate position data structure
        # this is still vulnerable to outlier/junk positions
        
        for sonar_idx in range(numsonars):

#            if sonar_idx < len(sonar_discontinuity_flag) and sonar_discontinuity_flag[sonar_idx]:
#                print("SONAR", sonar_idx, "/", numsonars, "->", "image", cam_minindex[sonar_idx])
#            else:
#                print(".....", sonar_idx, "/", numsonars, "->", "image", cam_minindex[sonar_idx])
                
            #print("image xy", imagex[cam_minindex[sonar_idx]], imagey[cam_minindex[sonar_idx]])
            sonarx[sonar_idx] = imagex[cam_minindex[sonar_idx]]
            sonary[sonar_idx] = imagey[cam_minindex[sonar_idx]]
            #print("x and y: ",sonar_idx, sonarx[sonar_idx],sonary[sonar_idx])
        # second pass to determine headings

        #for sonar_idx in range(numsonars):

            image_idx = cam_minindex[sonar_idx]
            leg_idx = compute_leg_idx_from_image_idx(image_idx)
            sonarx_leg_proj[sonar_idx], sonary_leg_proj[sonar_idx], _ = closest_point_on_line(leg_m_list[leg_idx], leg_c_list[leg_idx], sonarx[sonar_idx], sonary[sonar_idx])

            #print("x and y: ",sonar_idx, sonarx_leg_proj[sonar_idx],sonary_leg_proj[sonar_idx]) 
            sonardx[sonar_idx] = leg_dx_list[leg_idx]
            sonardy[sonar_idx] = leg_dy_list[leg_idx]
            #print("dx and dy: ",sonar_idx, sonardx[sonar_idx],sonardy[sonar_idx])

#            if sonar_idx==0 or (sonar_idx > 0 and sonar_discontinuity_flag[sonar_idx-1] == True):
#                dx = sonarx[sonar_idx+1] - sonarx[sonar_idx]
#                dy = sonary[sonar_idx+1] - sonary[sonar_idx]
#                #print("idx+1 y, idx y", sonary[sonar_idx+1], sonary[sonar_idx])
#            else:
#                dx = sonarx[sonar_idx] - sonarx[sonar_idx-1] 
#                dy = sonary[sonar_idx] - sonary[sonar_idx-1]
#            veclen = math.sqrt(dx*dx + dy*dy);
#            dx = dx/veclen;
#            dy = dy/veclen;
#            sonardx[sonar_idx] = dx
#            sonardy[sonar_idx] = dy

#            if sonar_idx < numsonars - 1:
#                print("heading", sonar_idx, sonardx[sonar_idx], sonardy[sonar_idx], sonar_discontinuity_flag[sonar_idx])
#            else:
#                print("LAST heading", sonar_idx, sonardx[sonar_idx], sonardy[sonar_idx])
            
            R_rect2utm = np.array([[sonardx[sonar_idx], -sonardy[sonar_idx], 0.0], \
                                   [sonardy[sonar_idx], sonardx[sonar_idx], 0.0], \
                                   [0.0, 0.0, 1.0]])
            #print(R_rect2utm)
            R_utm2rect = np.transpose(R_rect2utm)

            sonar_R_rect2utm_list.append(R_rect2utm)
            sonar_R_utm2rect_list.append(R_utm2rect)

        compute_sonar_leg_endpts()

    # no sonar2cam file?  ok, let's create one (hint: go get some coffee)...

    else:

#        print("should not be here")
#        sys.exit()
        
        # camera info...needed for syncing
    
        firstframe_capturetime, firstframe_datetime = get_frame_timeinfo(0)

#        sonar_superdir = "/data/scallops/sonar/20170817_IM/"

        f = open(sonar2cam_filename, "w")
        
        # this should be a directory traversal
        for sonar_idx in range(numsonars):

            sonar_idx_str = str(sonar_idx).zfill(3)
            sonarname = sonar_prefix + sonar_idx_str + "_MST_data.mat"
            filename = sonar_superdir + sonarname
            sonar_filename.append(filename)
            #print(filename)
            M = loadmat(filename)
            #    print(M["data"][0][0]['timeCorrelation']['dwSystime'][0][0][0][0])  
            year = M["data"][0][0]['timeCorrelation']['tm_year'][0][0][0][0]   #offset so add base? e.g. 2017 = 117  
            month = M["data"][0][0]['timeCorrelation']['tm_mon'][0][0][0][0]   #0-11
            mday = M["data"][0][0]['timeCorrelation']['tm_mday'][0][0][0][0]   #1-31
            hour = M["data"][0][0]['timeCorrelation']['tm_hour'][0][0][0][0]   
            min = M["data"][0][0]['timeCorrelation']['tm_min'][0][0][0][0]  
            sec = M["data"][0][0]['timeCorrelation']['tm_sec'][0][0][0][0]
            D = datetime(1900+year, 1+month, mday, hour, min, sec)
            sonar_datetime.append(D)

            image_mindiff = 100000000
            for image_idx in range(num_imageframes):
                idxframe_capturetime, idxframe_datetime = get_frame_timeinfo(image_idx)
                idxframe_capturetime_offset = idxframe_capturetime - firstframe_capturetime
                idxframe_datetime = firstframe_datetime + timedelta(seconds = idxframe_capturetime_offset)
                #    print(idxframe_capturetime)
                #            print(image_idx, idxframe_datetime)
                #    print(idxframe_capturetime_offset)
                absdiff = abs((sonar_datetime[sonar_idx] - idxframe_datetime).total_seconds())
                if absdiff < image_mindiff:
                    image_mindiff = absdiff
                    image_minindex = image_idx
                    
            print(sonar_idx, image_minindex, image_mindiff, sonar_datetime[sonar_idx].timestamp())
#            f.write(str(sonar_idx) + " " +  str(image_minindex) + " " + str(image_mindiff) + "\n")
            f.write(str(sonar_idx) + " " +  str(image_minindex) + " " + str(image_mindiff) + " " + str(sonar_datetime[sonar_idx].timestamp()) + "\n")
            f.flush()
            
        f.close()
        print("finished preprocess_sonar()")
        sys.exit()

    print("finished loading sonar!")
    #sys.exit()
    
#    return sonar_datetime, sonar_filename

#-------------------------------------------------------------------------------------

def get_image_filename(metadatafilename):
    filename_split = metadatafilename.split("/")
    imsuperdir = filename_split[-5] + "/"
    imsubdir = filename_split[-2] + "/"
    imname = filename_split[-1]
    fullimfilename =  impathprefix + imsuperdir + imsubdir + imname
    file_exists = path.exists(fullimfilename)
#    print(imsuperdir, imsubdir, imname)
    return imsuperdir, imsubdir, imname, file_exists

#-------------------------------------------------------------------------------------

def construct_npz_filename(imsubdir1, imname1, imsubdir2, imname2):
    return imsubdir1.replace("/", "_") + imname1[0:-4] + "_" + imsubdir2.replace("/", "_") + imname2[0:-4] + "_matches.npz"

#-------------------------------------------------------------------------------------

def takeY(elem):
    return elem[1]

#-------------------------------------------------------------------------------------

def load_sonar_image_from_mat(s_idx):

    sonar_idx_str = str(s_idx).zfill(3)
    sonarname = sonar_prefix + sonar_idx_str + "_MST_data.mat"
    filename = sonar_superdir + sonarname
    print(filename)
    S = loadmat(filename)

    sonar_left_im = np.reshape(S["data"][0][0]['leftChannel2'], (1000, 512))
    sonar_right_im = np.reshape(S["data"][0][0]['rightChannel2'], (1000, 512))

    sonar_left_nadir_edges, sonar_right_nadir_edges = detect_sonar_nadir_edges(sonar_left_im, sonar_right_im)
#    sonar_left_nadir_edges, sonar_right_nadir_edges = detect_sonar_nadir_edges(sonar_left_im, sonar_right_im)
#    slant_range_correction(sonar_left_im, sonar_right_im, sonar_left_nadir_edges, sonar_right_nadir_edges)
    
    sonar_left_im = np.fliplr(sonar_left_im)
#    sonar_left_im = cv2.flip(sonar_left_im, 1)

    sonar_im = np.hstack((sonar_left_im, sonar_right_im))
    sonar_nadir_mask_im = make_nadir_mask(sonar_im, sonar_left_nadir_edges, sonar_right_nadir_edges)
    
    return sonar_im, sonar_nadir_mask_im


#-------------------------------------------------------------------------------------
def load_rock_boxes(sonar_imname):

    load_retinanet()

    sonar_im = cv2.imread(sonar_imname, cv2.IMREAD_GRAYSCALE)
    print(sonar_imname)

    i = sonar_imname.find('images/')
    cur_sonar_name = sonar_imname[i+7:]
    print('cur sonar:', cur_sonar_name)

    for elem in annotation_list:
        elem = elem.split(',')
        if cur_sonar_name in elem[0]:
            x1 = int(elem[1])
            y1 = int(elem[2])
            x2 = int(elem[3])
            y2 = int(elem[4])
            sonar_im = cv2.rectangle(sonar_im, (x1,y1), (x2,y2), (255, 0, 0),2)

    return sonar_im
#-------------------------------------------------------------------------------------                                  
# vert flipping built in here
# not dealing with nadir mask here -- just sending copy of image as a placeholder

def load_sonar_image_from_png(s_idx):

    sonar_idx_str = str(s_idx).zfill(3)
    imagename = sonar_prefix + sonar_idx_str + "_MST_data.png"
    #print("sonar name: ", imagename)
    im_filename = sonar_superdir + "images/" + imagename
    #im_filename = new_sonar_superdir + imagename
    #sonar_im = cv2.imread(im_filename, cv2.IMREAD_GRAYSCALE)
    sonar_im = load_rock_boxes(im_filename)
    sonar_im = np.flipud(sonar_im)   # the big decision after days of dawning realization...!  added 7/3/2020
    

    #out_left_filename = "/tmp/LEFT_" + str(fixed_nadir_radius) + "_" + sonar_prefix + sonar_idx_str + "_MST_data.png"
    #out_right_filename = "/tmp/RIGHT_" + str(fixed_nadir_radius) + "_" + sonar_prefix + sonar_idx_str + "_MST_data.png"
    #print(out_left_filename, out_right_filename)

    #left_sonar_waterfall_im = sonar_waterfall_im[:,0:fixed_nadir_radius,:]
    #right_sonar_waterfall_im = sonar_waterfall_im[:, sonar_im_width-fixed_nadir_radius:,:]
    #horiz_left_sonar_waterfall_im = cv2.rotate(left_sonar_waterfall_im, cv2.ROTATE_90_CLOCKWISE)
    #horiz_right_sonar_waterfall_im = cv2.rotate(right_sonar_waterfall_im, cv2.ROTATE_90_CLOCKWISE)


#    sonar_left_nadir_edges, sonar_right_nadir_edges = detect_sonar_nadir_edges(sonar_left_im, sonar_right_im)
#    sonar_left_nadir_edges, sonar_right_nadir_edges = detect_sonar_nadir_edges(sonar_left_im, sonar_right_im)
#    slant_range_correction(sonar_left_im, sonar_right_im, sonar_left_nadir_edges, sonar_right_nadir_edges)
    
#    sonar_left_im = np.fliplr(sonar_left_im)
#    sonar_left_im = cv2.flip(sonar_left_im, 1)

#    sonar_im = np.hstack((sonar_left_im, sonar_right_im))
#    sonar_nadir_mask_im = make_nadir_mask(sonar_im, sonar_left_nadir_edges, sonar_right_nadir_edges)
    
    return sonar_im, sonar_im


#-------------------------------------------------------------------------------------
def load_sonar_image_list(s_idx, three_sonar_list):

    new_sonar_idx_str = str(s_idx).zfill(3)
    new_imagename = sonar_prefix + new_sonar_idx_str + "_MST_data.png"                         
                                     
    #new_im_filename = sonar_superdir + "images/" + new_imagename
    new_im_filename = new_sonar_superdir  + new_imagename
    three_sonar_list.append(new_im_filename)

    return three_sonar_list

def load_sonar_image_with_rock(list_idx):

    load_retinanet()
    global rock_im

    cur_sonar_path = three_sonar_list[list_idx]
    rock_im = cv2.imread(cur_sonar_path, cv2.IMREAD_GRAYSCALE)
    print(rock_im.shape)
    print(cur_sonar_path)

    i = cur_sonar_path.find('images/')
    cur_sonar_name = cur_sonar_path[i+7:]
    #print('cur sonar:', cur_sonar_name)
    
    for elem in annotation_list:
        elem = elem.split(',')
        if cur_sonar_name in elem[0]:
            x1 = int(elem[1])
            y1 = int(elem[2])
            x2 = int(elem[3])
            y2 = int(elem[4])
            rock_im = cv2.rectangle(rock_im, (x1,y1), (x2,y2), (255, 0, 0),1)

    return rock_im


#-------------------------------------------------------------------------------------

# -1 return value means "leg not found"

def compute_leg_idx_from_image_idx(image_idx):
    for i in range(len(leg_end_idx_list)):
        if image_idx <= leg_end_idx_list[i]:
            #print("computed leg", image_idx, i, leg_end_idx_list[i])  
            return i
    return -1

#-------------------------------------------------------------------------------------

def clip_slope_intercept_line_vs_mosaic_bbox(leg_idx):
    m = leg_m_list[leg_idx]
    c = leg_c_list[leg_idx]

#    print("clipping leg", leg_idx)
#    print(m, c)

#    print("utm left/right", mosaic_utm_x_left, mosaic_utm_x_right)
#    print("utm top/bottom", mosaic_utm_y_top, mosaic_utm_y_bottom)

    inv_m = 1.0 / m
    inv_c = -c / m

    x_top = inv_m * mosaic_utm_y_top + inv_c
    x_bot = inv_m * mosaic_utm_y_bottom + inv_c

    return x_top, x_bot

#    else:
#        return -1, -1
        

#    sys.exit()
    
#-------------------------------------------------------------------------------------

# taking globals imxtrans, yxtrans
# pixels_per_meter is global based on imscale

def refresh_transformed_images(imscale, src_im_list, H_list, sonar_idx_list, sonar_im_list, sonar_nadir_mask_im_list):

    # corners of mosaic image in UTM coords -- needed for drawing RANSAC line fits
    # note that +pixel y is down, whereas +UTM y is up
    
    mpp = 1.0 / pixels_per_meter
    global mosaic_utm_x_left, mosaic_utm_x_right, mosaic_utm_y_top, mosaic_utm_y_bottom  
    mosaic_utm_x_left = imagex[context_index] - 0.5 * mpp * float(mosaic_window_width)
    mosaic_utm_x_right = imagex[context_index] + 0.5 * mpp * float(mosaic_window_width)
    mosaic_utm_y_top = imagey[context_index] + 0.5 * mpp * float(mosaic_window_height)
    mosaic_utm_y_bottom = imagey[context_index] - 0.5 * mpp * float(mosaic_window_height)

    
    # correcting scale
    H_scale = np.array([[imscale, 0.0, 0.0],
                        [0.0, imscale, 0.0], 
                        [0.0, 0.0, 1.0]])


    # correcting translation 
#    H_translate = np.array([[1.0, 0.0, (1.0/imscale)*imwidth/4],
#                            [0.0, 1.0, (1.0/imscale)*imheight/2], 
#                            [0.0, 0.0, 1.0]])

    H_translate = np.array([[1.0, 0.0, imxtrans+mosaic_window_width/2-imscale*imwidth/2],
                        [0.0, 1.0, imytrans+mosaic_window_height/2-imscale*imheight/2], 
                        [0.0, 0.0, 1.0]])


#    H_mosaic = np.matmul(H_scale, H_translate)
    H_mosaic = np.matmul(H_translate, H_scale)

    dst_im_scaled_list = []
    dst_im_mask_scaled_list = []
    
    for i in range(len(H_list)):

#        dst_im = cv2.warpPerspective(src_im_list[i], np.matmul(H_mosaic, H_list[i]), (mosaic_window_width, mosaic_window_height))
        dst_im_scaled = cv2.warpPerspective(src_im_list[i], np.matmul(H_mosaic, H_list[i]), (mosaic_window_width, mosaic_window_height))
#        dst_im_scaled = cv2.resize(dst_im, (0, 0), fx=imscale, fy=imscale)
        
#        if i == 0:
#            im_mosaic = np.zeros(dst_im.shape, np.uint8)

        dst_im_mask_scaled = cv2.warpPerspective(mask_im, np.matmul(H_mosaic, H_list[i]), (mosaic_window_width, mosaic_window_height))
#        dst_im_mask = cv2.warpPerspective(mask_im, np.matmul(H_mosaic, H_list[i]), (mosaic_window_width, mosaic_window_height))
#        dst_disttrans = cv2.distanceTransform(dst_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
#        dst_disttrans_display = cv2.normalize(dst_disttrans, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
#        dst_disttrans_display_scaled = cv2.resize(dst_disttrans, (0, 0), fx=0.5, fy=0.5)
#        dst_list.append(dst)
#        dst_mask_list.append(dst_disttrans)

#        dst_im_mask_scaled = cv2.resize(dst_im_mask, (0, 0), fx=imscale, fy=imscale)

        dst_im_scaled_list.append(dst_im_scaled)
        dst_im_mask_scaled_list.append(dst_im_mask_scaled)

    # sonar

    if do_sonar:
        
        # first take waterfall image corners to rect corners
        # (0, 0) -> (0, sonarrectwidth/2)
        # (waterfall_im_width - 1, 0) -> (0, -sonarrectwidth/2)
        # (waterfall_im_width - 1, waterfall_im_height - 1) -> (-sonarrectlength, -sonarrectwidth/2)
        # (0, waterfall_im_height) -> (-sonarrectlength, sonarrectwidth/2)

        # compute the damn homography

 #       waterfallcorners = np.array([[0, sonar_im_width - 1, sonar_im_width - 1,  0], \
 #                                    [0, 0,                  sonar_im_height - 1, sonar_im_height - 1], \
 #                                    [1, 1, 1, 1]])
        #print(waterfallcorners)
        #sys.exit()
        #    waterfallcornerslist = np.array([[0, 0], [sonar_im_width - 1, 0], [sonar_im_width - 1, sonar_im_height - 1], [0, sonar_im_height - 1]]) 
        #    sonarcornerslist = np.transpose(sonarcorners)[:,0:2]
        #    print(np.transpose(waterfallcorners)[:,0:2])
#        (H_waterfall_to_rect, _) = cv2.findHomography(np.transpose(waterfallcorners)[:,0:2], np.transpose(sonarcorners)[:,0:2])
        #    print(H_waterfall_to_rect)
#        waterfall_corners_prime = np.matmul(H_waterfall_to_rect, waterfallcorners)
        #    print(waterfall_corners_prime)
        #    sys.exit()

        # then take rect corners to UTM

        if math.isnan(imagedx[context_index]) or math.isnan(imagedy[context_index]) or not do_heading_rotation:
            # instead of punting, just make rotation the identity matrix
            Tr = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        else:            
            heading_rads = math.atan2(imagedy[context_index], imagedx[context_index])
            heading_degs = math.degrees(heading_rads)
            #print("1 degs", heading_degs)

            Tr = np.array([[math.cos(-heading_rads + math.pi/2), -math.sin(-heading_rads + math.pi/2), 0.0],
                           [math.sin(-heading_rads + math.pi/2), math.cos(-heading_rads + math.pi/2), 0.0], 
                           [0.0, 0.0, 1.0]])

        Ts = np.array([[pixels_per_meter, 0.0, 0.0],
                       [0.0, -pixels_per_meter, 0.0], 
                       [0.0, 0.0, 1.0]])
        Tt = np.array([[1.0, 0.0, -imagex[context_index]],
                       [0.0, 1.0, -imagey[context_index]], 
                       [0.0, 0.0, 1.0]])
#        T_xform = np.matmul(Ts, Tt)
        T_xform = Ts @ Tr @ Tt

        T_final = np.array([[1.0, 0.0, imxtrans+mosaic_window_width/2],
                            [0.0, 1.0, imytrans+mosaic_window_height/2], 
                            [0.0, 0.0, 1.0]])

        dst_sonar_im_scaled_list = []
        dst_sonar_nadir_mask_im_scaled_list = []
    
        for i in range(len(sonar_idx_list)):
            #print(len(sonar_idx_list))
            j = sonar_idx_list[i]
            #print(i,j)

            # get sonar rectangle corners in the mosaic image

            sonar_T_rect2utm = np.array([[1.0, 0.0, sonarx_leg_proj[j] + sonar_alignment_factor_list[i] * sonardx[j]],
                                         [0.0, 1.0, sonary_leg_proj[j] + sonar_alignment_factor_list[i] * sonardy[j]], 
                                         [0.0, 0.0, 1.0]])
#            sonar_T_rect2utm = np.array([[1.0, 0.0, sonarx_leg_proj[j]],
#                                         [0.0, 1.0, sonary_leg_proj[j]], 
#                                         [0.0, 0.0, 1.0]])
#            sonar_T_rect2utm = np.array([[1.0, 0.0, sonarx[j]],
#                                         [0.0, 1.0, sonary[j]], 
#                                         [0.0, 0.0, 1.0]])

            sonar_TR_rect2utm = np.matmul(sonar_T_rect2utm, sonar_R_rect2utm_list[j])
            T_waterfall_to_utm = np.matmul(sonar_TR_rect2utm, H_waterfall_to_rect)
            T_combined = np.matmul(T_xform, T_waterfall_to_utm)
            T_waterfall_to_image = np.matmul(T_final, T_combined)
            #print("waterfall", T_waterfall_to_image)
            #       image_sonarcorners = np.matmul(T_waterfall_to_image, waterfallcorners)
            #        print(image_sonarcorners)
        
            #for corner_idx in range(4):
            #    sonimcorns[corner_idx][0] = imxtrans+image_sonarcorners[0][corner_idx] + mosaic_window_width/2
            #    sonimcorns[corner_idx][1] = imytrans+image_sonarcorners[1][corner_idx] + mosaic_window_height/2
            dst_sonar_im_scaled = cv2.warpPerspective(sonar_im_list[i], T_waterfall_to_image, (mosaic_window_width, mosaic_window_height))
            #print(sonar_im_list[i])
            dst_sonar_im_scaled_list.append(cv2.cvtColor(dst_sonar_im_scaled, cv2.COLOR_GRAY2BGR))
            #cv2.imshow("waterfall", cv2.cvtColor(dst_sonar_im_scaled, cv2.COLOR_GRAY2BGR))
            dst_sonar_nadir_mask_im_scaled = cv2.warpPerspective(sonar_nadir_mask_im_list[i], T_waterfall_to_image, (mosaic_window_width, mosaic_window_height))
            dst_sonar_nadir_mask_im_scaled_list.append(cv2.cvtColor(dst_sonar_nadir_mask_im_scaled, cv2.COLOR_GRAY2BGR))
    else:
        dst_sonar_im_scaled_list = []
        dst_sonar_nadir_mask_im_scaled_list = []
        
    # done
    
    return dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list

#-------------------------------------------------------------------------------------

# image_in_* variables are globals, so I'm going to be lazy and not return them

def dumb_update_sonar_for_new_image(init_idx):
    
    image_in_sonarrect_idx_list.clear()
    image_in_sonarrect_waterfall_im_list.clear()
    image_in_sonarrect_waterfall_nadir_mask_im_list.clear()
    # this small index indicates which sonar waterfall will be "featured" in the mosaic display
    current_sonarrect_idx = 0

    # j is a big number index into the global sonar data structure
    
    for j in range(len(sonarx)):
        rectimage = np.matmul(sonar_R_utm2rect_list[j], np.array([[imagex[init_idx]-sonarx[j]],[imagey[init_idx]-sonary[j]], [1.0]]))
        print("rectimage: ", rectimage)
        if sonar_rectpath.contains_point([rectimage[0][0], rectimage[1][0]]):
            image_in_sonarrect_idx_list.append(j)
            sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_mat(j)
            image_in_sonarrect_waterfall_im_list.append(sonar_waterfall_im)
            image_in_sonarrect_waterfall_nadir_mask_im_list.append(sonar_waterfall_nadir_mask_im)

#-------------------------------------------------------------------------------------

# have not tested the case that the image overlaps NO sonars ... pretty sure something
# bad will happen downstream if this occurs

# init_idx is global camera frame index...damn my naming conventions are shit here

def smart_update_sonar_for_new_image(init_idx):
    
    # this small index indicates which sonar waterfall will be "featured" in the mosaic display
    current_sonarrect_idx = 0

    # j is a big number index into the global sonar data structure
    
    for j in range(len(sonarx)):

        # was this sonar previously in the overlap list?
        
        try:
            j_index = image_in_sonarrect_idx_list.index(j)
        except ValueError:
            j_index = -1

        # transform image center to "sonar rectangle coords" for THIS sonar
        
        rectimage = np.matmul(sonar_R_utm2rect_list[j], np.array([[imagex[init_idx]-sonarx[j]],[imagey[init_idx]-sonary[j]], [1.0]]))

        # is this image center inside the sonar rectangle?

        if sonar_rectpath.contains_point([rectimage[0][0], rectimage[1][0]]):

            # ADD a new overlapping swath to list because it's not already there
            if j_index < 0:
                image_in_sonarrect_idx_list.append(j)
#                sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_mat(j)
                sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_png(j)
                image_in_sonarrect_waterfall_im_list.append(sonar_waterfall_im)
                image_in_sonarrect_waterfall_nadir_mask_im_list.append(sonar_waterfall_nadir_mask_im)

        # REMOVE old swath that no longer overlaps from list
        elif j_index >= 0:
            image_in_sonarrect_idx_list.pop(j_index)
            image_in_sonarrect_waterfall_im_list.pop(j_index)
            image_in_sonarrect_waterfall_nadir_mask_im_list.pop(j_index)

#-------------------------------------------------------------------------------------

# are these two sonar images from the same leg?

def same_leg_sonar(sonar_idx1, sonar_idx2):
    if sonar_idx1 == sonar_idx2:
        return True
    #print("sonar idx")
    #print(sonar_idx1, sonar_idx2)
    image_idx1 = cam_minindex[sonar_idx1]
    image_idx2 = cam_minindex[sonar_idx2]
    #print("image idx")
    #print(image_idx1, image_idx2)
    leg_idx1 = compute_leg_idx_from_image_idx(image_idx1)
    leg_idx2 = compute_leg_idx_from_image_idx(image_idx2)
    #print("leg idx")
    #print(leg_idx1, leg_idx2)
    return leg_idx1 == leg_idx2

#-------------------------------------------------------------------------------------

# init_idx is CURRENT global camera frame index...damn my naming conventions are shit here
# side effect (and only purpose) of this function is to set image_in_sonarrect_idx_list,
# image_in_sonarrect_waterfall_im_list, and image_in_sonarrect_waterfall_nadir_mask_im_list

# this version will only use sonar2cam data structure to get ONE sonar image, no geometric transformations

def supersmart_update_sonar_for_new_image(init_idx):

    # j is a big number index into the global sonar data structure

    #print("cam index ", init_idx)
    
    #for j in range(len(sonarx)):
    #    print(j, cam_minindex[j])

    global sonar_context_index, current_sonarrect_idx
    global three_sonar_list
    three_sonar_list = []
    #print("sonar idx: ", sonar_context_index)
    #print("sonar idx: ", current_sonarrect_idx)

    new_sonar_context_index = compute_sonar_context_index(init_idx)
    #print("old sonar, new sonar", sonar_context_index, new_sonar_context_index)
    
    # new image
    if sonar_context_index == -1:
        pass
        #print("start")

    # no change
    elif new_sonar_context_index == sonar_context_index:
        #print("stasis -- don't do a thing")
        return

    # increment
#    elif new_sonar_context_index == sonar_context_image + 1:
#        print("increment")
#        sonar_context_index = new_sonar_context_index
#        sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_png(sonar_context_index + 1)
#        # delete old stuff from front of list if we have 3 already
#        if len(image_in_sonarrect_idx_list) == 3:
#            image_in_sonarrect_idx_list.pop(0)
#            image_in_sonarrect_waterfall_im_list.pop(0)
#            image_in_sonarrect_waterfall_nadir_mask_im_list.pop(0)
#        # append new stuff 
#        image_in_sonarrect_idx_list.append(sonar_context_index)
#        image_in_sonarrect_waterfall_im_list.append(sonar_waterfall_im)
#        image_in_sonarrect_waterfall_nadir_mask_im_list.append(sonar_waterfall_nadir_mask_im)

#        current_sonarrect_idx = 1   # but maybe not?
#        return
    
    # decrement
#    elif new_sonar_context_index == sonar_context_image - 1:
#        print("decrement")
#        sonar_context_index = new_sonar_context_index
#        sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_png(sonar_context_index - 1)
#        # delete old stuff from end of list if we have 3 already
#        if len(image_in_sonarrect_idx_list) == 3:
#            image_in_sonarrect_idx_list.pop(2)
#            image_in_sonarrect_waterfall_im_list.pop(2)
#            image_in_sonarrect_waterfall_nadir_mask_im_list.pop(2)
#        # list.insert(0, new stuff)
#        image_in_sonarrect_idx_list.insert(0, sonar_context_index)
#        image_in_sonarrect_waterfall_im_list.insert(0, sonar_waterfall_im)
#        image_in_sonarrect_waterfall_nadir_mask_im_list.insert(0, sonar_waterfall_nadir_mask_im)
#        current_sonarrect_idx = 1   # but maybe not?
#        return
    
    # jump, treat like we are starting fresh
    else:
        #print("jump to", new_sonar_context_index)
        image_in_sonarrect_idx_list.clear()
        image_in_sonarrect_waterfall_im_list.clear()
        image_in_sonarrect_waterfall_nadir_mask_im_list.clear()
        sonar_alignment_factor_list.clear()
        three_sonar_list.clear()
        
    sonar_context_index = new_sonar_context_index

    # try to load full list -- this covers start and jump
    
    for delta_index in range(-1, 2):
 #       print(delta_index)
        if new_sonar_context_index + delta_index >= 0 and new_sonar_context_index + delta_index < numsonars and \
           same_leg_sonar(new_sonar_context_index, new_sonar_context_index + delta_index):
            image_in_sonarrect_idx_list.append(new_sonar_context_index + delta_index)
            sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_png(new_sonar_context_index + delta_index)

            three_sonar_list = load_sonar_image_list((new_sonar_context_index + delta_index), three_sonar_list)
            #rock_im = load_sonar_image_with_rock(current_sonarrect_idx)

            image_in_sonarrect_waterfall_im_list.append(sonar_waterfall_im)
            image_in_sonarrect_waterfall_nadir_mask_im_list.append(sonar_waterfall_nadir_mask_im)
            
    # set current_sonarrect_idx based on what just happened

    if sonar_context_index == 0 or not same_leg_sonar(sonar_context_index, sonar_context_index - 1):
#        print("leg start, setting current to 0")
        current_sonarrect_idx = 0
        #print("beginning", len(image_in_sonarrect_idx_list))
        sonar_alignment_factor_list.append(0.0)
        sonar_alignment_factor_list.append(compute_sonar_alignment_factor(sonar_context_index, sonar_context_index + 1))

    else: #if sonar_context_index == numsonars - 1 or anything else
#        print("leg middle or end, setting current to 1")
        current_sonarrect_idx = 1
        if len(image_in_sonarrect_idx_list) == 2:
            #print("end", len(image_in_sonarrect_idx_list))
            sonar_alignment_factor_list.append(compute_sonar_alignment_factor(sonar_context_index, sonar_context_index - 1))
            sonar_alignment_factor_list.append(0.0)
        else:
            #print("middle", len(image_in_sonarrect_idx_list))
            sonar_alignment_factor_list.append(compute_sonar_alignment_factor(sonar_context_index, sonar_context_index - 1))
            sonar_alignment_factor_list.append(0.0)
            sonar_alignment_factor_list.append(compute_sonar_alignment_factor(sonar_context_index, sonar_context_index + 1))


#    sys.exit()


    #print("supersmart c->s", init_idx, sonar_context_index)

    #if len(image_in_sonarrect_idx_list) > 0 and image_in_sonarrect_idx_list[1] != sonar_context_index:
     #   print("popping!")
     #   image_in_sonarrect_idx_list.pop(0)
     #   image_in_sonarrect_waterfall_im_list.pop(0)
     #   image_in_sonarrect_waterfall_nadir_mask_im_list.pop(0)
       # print("need to pop!")
        
    #if len(image_in_sonarrect_idx_list) == 0:
        #print("appending!")
        #image_in_sonarrect_idx_list.append(sonar_context_index)
        #sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_png(sonar_context_index)
        #image_in_sonarrect_waterfall_im_list.append(sonar_waterfall_im)
        #image_in_sonarrect_waterfall_nadir_mask_im_list.append(sonar_waterfall_nadir_mask_im)

     #   print("appending")
     #   current_sonarrect_idx = 0

     #   if sonar_context_index > 0:
     #       image_in_sonarrect_idx_list.append(sonar_context_index-1)
     #       sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_png(sonar_context_index-1)
     #       image_in_sonarrect_waterfall_im_list.append(sonar_waterfall_im)
     #       image_in_sonarrect_waterfall_nadir_mask_im_list.append(sonar_waterfall_nadir_mask_im)
     #       current_sonarrect_idx = 1

#        image_in_sonarrect_idx_list.append(sonar_context_index)
#        sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_png(sonar_context_index)
#        image_in_sonarrect_waterfall_im_list.append(sonar_waterfall_im)
#        image_in_sonarrect_waterfall_nadir_mask_im_list.append(sonar_waterfall_nadir_mask_im)

 #       if sonar_context_index < numsonars - 1:
 #           image_in_sonarrect_idx_list.append(sonar_context_index+1)
 #           sonar_waterfall_im, sonar_waterfall_nadir_mask_im = load_sonar_image_from_png(sonar_context_index+1)
 #           image_in_sonarrect_waterfall_im_list.append(sonar_waterfall_im)
 #           image_in_sonarrect_waterfall_nadir_mask_im_list.append(sonar_waterfall_nadir_mask_im)
        
    #print("----------------------------------------------")
    
#    sys.exit()

#-------------------------------------------------------------------------------------

# this gets called every time we have a new "focus" camera image to center the mosaic on

def initialize_context_image(init_idx, imscale):

    print("prev and next neighbors", init_idx, image_prev_neighbor_idx[init_idx], image_next_neighbor_idx[init_idx])
    
    global leg_context_index
    #print("leg was", leg_context_index)
    leg_context_index = compute_leg_idx_from_image_idx(init_idx)
    #print("leg is NOW", leg_context_index)

    # utm coords of window
    
    #mosaic_utm_x_min
    # update name for new image
    
    imsuperdir, src_imsubdir, src_imname, src_file_found_flag = get_image_filename(camera['filename'][init_idx][0][0])
    if src_file_found_flag == False:
        print("cannot find image", imsuperdir, src_imsubdir, src_imname)
#        sys.exit()

    # sonar

    if do_sonar:
        #dumb_update_sonar_for_new_image(init_idx)
        #ns_before = time.time_ns()
        supersmart_update_sonar_for_new_image(init_idx)
        #ns_after = time.time_ns()
        #ns_diff = ns_after - ns_before
        #print("smart sonar: ", ns_diff)
        
    # update geometry related to new camera image
    
#    ns_before = time.time_ns()

    H_list = []
    imfilename_list = []
    
    imcenter_list = []
    imcorners_list = []

    H_list.append(np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0], 
                            [0.0, 0.0, 1.0]]))
    if src_file_found_flag:
        imfilename_list.append(impathprefix + imsuperdir + src_imsubdir + src_imname)
        print('imfilename:', impathprefix + imsuperdir + src_imsubdir + src_imname)
    else:
        imfilename_list.append("/data/scallops/images/notfound.jpg")
        
    encoded_imcenter = np.copy(imcenter)
    encoded_imcenter[2] = 0
    imcenter_list.append(np.transpose(encoded_imcenter))
    imcorners_list.append(transposed_imcorners)

    one_idx = 1

    idx_list = []
    idx_list.append(init_idx)
    imsubdir_list = []
    imsubdir_list.append(src_imsubdir)
    imname_list = []
    imname_list.append(src_imname)
    
    # iterate over neighbors...note that all_nodes[init_idx - start_index][0] is the source node
    
    for idx in all_nodes[init_idx - start_index][1:]:
        imsuperdir, dst_imsubdir, dst_imname, dst_file_found_flag = get_image_filename(camera['filename'][idx][0][0])
        if dst_file_found_flag == False:
            print("cannot find neighbor image", imsuperdir, dst_imsubdir, dst_imname)
            continue

        # order might be switched -- check for both
        print("src_imsubdir: ",src_imsubdir)
        print("src_imname: ", src_imname)
        print("dst_imsubidr: ", dst_imsubdir)
        print("dst_imname: ",dst_imname)
        npz_filename = construct_npz_filename(src_imsubdir, src_imname, dst_imsubdir, dst_imname)
        reverse_npz_filename = construct_npz_filename(dst_imsubdir, dst_imname, src_imsubdir, src_imname)
        npzdirectory = "/data/scallops/superglue_output/" + imsuperdir
        npzfullfilename = npzdirectory + npz_filename
        reverse_npzfullfilename = npzdirectory + reverse_npz_filename
        npz_exists = path.exists(npzfullfilename)
        reverse_npz_exists = path.exists(reverse_npzfullfilename)
        if npz_exists:
            H = superglue.compute_overlap(npzfullfilename, debug=False, num_matches_threshold=20)
            if H is None:
                continue
#            print(npz_filename, "FOUND")
            H = np.linalg.inv(H)
            H_list.append(H)
            imfilename_list.append(impathprefix + imsuperdir + dst_imsubdir + dst_imname)
            idx_list.append(idx)
            imsubdir_list.append(dst_imsubdir)
            imname_list.append(dst_imname)
        elif reverse_npz_exists:
            H = superglue.compute_overlap(reverse_npzfullfilename, debug=False, num_matches_threshold=20)
            if H is None:
                continue
#            print(reverse_npz_filename, "FOUND REVERSE")
            # do I need to invert H because of reversal of direction???
            H_list.append(H)
            imfilename_list.append(impathprefix + imsuperdir + dst_imsubdir + dst_imname)
            idx_list.append(idx)
            imsubdir_list.append(dst_imsubdir)
            imname_list.append(dst_imname)
        else:
            print(npzfullfilename, reverse_npzfullfilename, "DO NOT EXIST")
            continue

        print("imlist: ",imname_list)
        xformed_imcorners = np.transpose(np.matmul(H, imcorners))    
        xformed_imcenter = np.transpose(np.matmul(H, imcenter))

        encoded_imcenter = np.copy(xformed_imcenter)
        encoded_imcenter[2] = one_idx
        imcenter_list.append(encoded_imcenter)
    
        imcorners_list.append(np.copy(xformed_imcorners))

        for j in range(4):
            xformed_imcorners[j][1] = imheight-1-xformed_imcorners[j][1]

        one_idx += 1
    
    y_sorted_imcenter_list = sorted(imcenter_list, key=takeY)
    ordered_indices = []
    for i in range(len(y_sorted_imcenter_list)):
        idx = int(y_sorted_imcenter_list[i][2])
        ordered_indices.append(idx)
        if idx == 0:
            current_idx = i

    src_im_list = []

#    ns_after = time.time_ns()
#    ns_diff = ns_after - ns_before
#    print("middle section: ", ns_diff)

#    ns_before = time.time_ns()

    for i in range(len(H_list)):
        src_im = cv2.imread(imfilename_list[i])
        src_im_list.append(src_im)

    # this is where the image file actually gets loaded and various warps are performed

    dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)
    #print("dst_im_scaled_list: ",dst_im_scaled_list)
#    ns_after = time.time_ns()
#    ns_diff = ns_after - ns_before
#    print("image handling: ", ns_diff)
#    print("---------------------------------------------------------")
    
    return imsuperdir, imsubdir_list, imname_list, idx_list, src_im_list, H_list, dst_im_scaled_list, dst_im_mask_scaled_list, ordered_indices, current_idx, imcenter_list, imcorners_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list

#-------------------------------------------------------------------------------------

# idx's in idx_list are into the camera data structure -- big numbers
# current_idx is into idx_list -- it's a small number

def build_PLL_corners_list(current_idx, idx_list):

#    print("**********************************************")
#    print(current_idx)
#    print(idx_list)

    utmlines = []

    utm_array_list = []

    # H from current_idx to all images (including itself) in UTM coords
    
    for t in idx_list:
        p0 = np.array(utm.from_latlon(camera['PLL'][0][0][t], camera['PLL'][0][1][t])[0:2])
        p1 = np.array(utm.from_latlon(camera['PLL'][1][0][t], camera['PLL'][1][1][t])[0:2])
        p2 = np.array(utm.from_latlon(camera['PLL'][2][0][t], camera['PLL'][2][1][t])[0:2])
        p3 = np.array(utm.from_latlon(camera['PLL'][3][0][t], camera['PLL'][3][1][t])[0:2])
        utm_array_list.append(np.array([p0, p1, p2, p3]))

        utmlines.append([p0[0:2], p1[0:2]])
        utmlines.append([p1[0:2], p2[0:2]])
        utmlines.append([p2[0:2], p3[0:2]])
        utmlines.append([p3[0:2], p0[0:2]])

    H_utm2utm_list = []
    for t in range(len(idx_list)):
#        print(utm_array_list[0])
#        sys.exit()
        (H_utm2utm, _) = cv2.findHomography(utm_array_list[current_idx], utm_array_list[t])
        H_utm2utm_list.append(H_utm2utm)

    # H from zero image corners (NOT transformed) to UTM corners for all images

    H_im2utm_list = []
    for t in range(len(utm_array_list)):
#        print(t)
#        print(corners_list[0][:,0:2])
#        print(transposed_pll_imcorners[:,0:2])
#        print(utm_array_list[t])
        (H_im2utm, _) = cv2.findHomography(transposed_pll_imcorners[:,0:2], utm_array_list[t])
#        print(H_im2utm)
        H_im2utm_list.append(H_im2utm)

#    lc = mc.LineCollection(utmlines)  #, colors=c, linewidths=2)
##    plt.clf()
#    plt_ax.add_collection(lc)
#    #ax.scatter(plot_imagex[startindex:stopindex], plot_imagey[startindex:stopindex], c=(1,0,0), s=20)
#    #ax.autoscale()
#    plt_ax.axis('equal')
#    #ax.margins(0.1)
#    plt.show()
#    plt.pause(0.001)
    
    return H_utm2utm_list, H_im2utm_list

#-------------------------------------------------------------------------------------

# current_sonarrect_idx is a small index into image_in_sonarrect_idx_list, rather than a big index of the global sonar list

# current_idx is not the global index of the context image -- it's the index into a small list of the highlighted image, which can
# be either the context image or one of its neighbors
# idx is the global index of current_idx

def draw_normal_overlays(draw_im, imsubdir_list, imname_list, ordered_indices, current_idx, imcenter_list, imcorners_list, idx_list, image_in_sonarrect_idx_list, current_sonarrect_idx):

     
#    print(far_next_set)
#    print(leg_end_set)
#    neither_set = far_next_set.symmetric_difference(leg_end_set)
#    print(neither_set)
#    sys.exit()
    
#    ns_before = time.time_ns()

    idx = idx_list[ordered_indices[current_idx]]
    cv2.putText(draw_im, imsubdir_list[ordered_indices[current_idx]] + imname_list[ordered_indices[current_idx]], (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    cv2.putText(draw_im, str(round(imagex[idx], 2)) + " " + str(round(imagey[idx], 2)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    cv2.putText(draw_im, str(round(imagedx[idx], 2)) + " " + str(round(imagedy[idx], 2)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    cv2.putText(draw_im, str(round(imscale_factor, 3)), (10, mosaic_window_height - 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    cv2.putText(draw_im, str(round(imxtrans, 2)) + " " + str(round(imytrans, 2)), (10, mosaic_window_height - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    cv2.putText(draw_im, str(leg_context_index), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

    # camera images
    
    for i in range(len(imcenter_list)):
        print("imcenter: ",imcenter_list)
        if i == ordered_indices[current_idx]:
            draw_thickness = 2
            draw_color = (0, 0, 255)

            if image_farnext[idx] == 1:
                cv2.circle(draw_im, \
                           (int(imxtrans+imscale_factor*(imcenter_list[i][0]-imwidth/2) + mosaic_window_width/2), \
                            int(imytrans+imscale_factor*(imcenter_list[i][1]-imheight/2) + mosaic_window_height/2)), \
                           int(imscale_factor*16), (255, 255, 255), draw_thickness)

        else:
            draw_thickness = 1
            draw_color = (0, 255, 255)
 
        # label
        
        cv2.putText(draw_im, str(idx_list[i]), \
                    (int(imxtrans+imscale_factor*(imcorners_list[i][0][0]-imwidth/2)+mosaic_window_width/2), \
                     int(imytrans+imscale_factor*(imcorners_list[i][0][1]-imheight/2)+mosaic_window_height/2) - 5), \
                    cv2.FONT_HERSHEY_PLAIN, 1.0, draw_color, 1)

        # neighbor centers and corners

        cv2.circle(draw_im, \
                   (int(imxtrans+imscale_factor*(imcenter_list[i][0]-imwidth/2) + mosaic_window_width/2), \
                    int(imytrans+imscale_factor*(imcenter_list[i][1]-imheight/2) + mosaic_window_height/2)), \
                   int(imscale_factor*8), draw_color, draw_thickness)
        xlist = []
        ylist = []
        for j in range(4):
            cv2.line(draw_im, \
                     (int(imxtrans+imscale_factor*(imcorners_list[i][j][0]-imwidth/2)+mosaic_window_width/2), \
                      int(imytrans+imscale_factor*(imcorners_list[i][j][1]-imheight/2)+mosaic_window_height/2)), \
                     (int(imxtrans+imscale_factor*(imcorners_list[i][(j+1)%4][0]-imwidth/2)+mosaic_window_width/2), \
                      int(imytrans+imscale_factor*(imcorners_list[i][(j+1)%4][1]-imheight/2)+mosaic_window_height/2)), \
                     tuple(c/5 for c in draw_color), draw_thickness)
            xlist.append(int(imxtrans+imscale_factor*(imcorners_list[i][j][0]-imwidth/2)+mosaic_window_width/2))
            xlist.append(int(imxtrans+imscale_factor*(imcorners_list[i][(j+1)%4][0]-imwidth/2)+mosaic_window_width/2))
            ylist.append(int(imytrans+imscale_factor*(imcorners_list[i][j][1]-imheight/2)+mosaic_window_height/2))
            ylist.append(int(imytrans+imscale_factor*(imcorners_list[i][(j+1)%4][1]-imheight/2)+mosaic_window_height/2))
            print("line: ", int(imxtrans+imscale_factor*(imcorners_list[i][j][0]-imwidth/2)+mosaic_window_width/2),int(imxtrans+imscale_factor*(imcorners_list[i][(j+1)%4][0]-imwidth/2)+mosaic_window_width/2))
        img = cv2.imread(impathprefix + imsuperdir + imsubdir_list[i] +  imname_list[i])
        print(imsubdir_list[i] +  imname_list[i])
        dim = (int(img.shape[1]*imscale_factor),int(img.shape[0]*imscale_factor))
        img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
        print(min(ylist),max(ylist))
        print(min(xlist),max(xlist))
        #draw_im[min(ylist):min(ylist)+img.shape[1],min(xlist):min(xlist)+img.shape[0]] += img
        #draw_im[min(xlist):min(xlist)+img.shape[0],min(ylist):min(ylist)+img.shape[1]] += img
        print("drawoverlay: ", imname_list)
    # UTM coords of other images in this dataset

    if math.isnan(imagedx[idx]) or math.isnan(imagedy[idx]):
        cv2.putText(draw_im, "NO HEADING", (mosaic_window_width - 100, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        return
    
    heading_rads = math.atan2(imagedy[context_index], imagedx[context_index])
    heading_degs = math.degrees(heading_rads)
    
#        pt = np.array([imagedx[idx], imagedy[idx], 1.0])
    pt = np.array([0.0, 1.0, 1.0])   # true north
    p = np.transpose(pt)

    if not do_heading_rotation:
        Tr = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    else:
        Tr = np.array([[math.cos(-heading_rads + math.pi/2), -math.sin(-heading_rads + math.pi/2), 0.0],
                       [math.sin(-heading_rads + math.pi/2), math.cos(-heading_rads + math.pi/2), 0.0], 
                       [0.0, 0.0, 1.0]])

    ppix = np.matmul(Tr, p)

    # compass overlay for heading indication -- really does not belong here
    
    compass_radius = 25.0
    cv2.circle(draw_im, (mosaic_window_width - 50, 75), int(compass_radius), (255, 255, 255), 1)
    cv2.line(draw_im, \
             (mosaic_window_width - 50, 75), \
             (int(mosaic_window_width - 50 + compass_radius*ppix[0]), int(75 - compass_radius*ppix[1])), \
             (255, 255, 255), 1)
        
    heading_degs_str = str(round(heading_degs, 1))
    cv2.putText(draw_im, heading_degs_str, (mosaic_window_width - 100, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    cv2.putText(draw_im, str(round(ppix[0], 2)) + ", " + str(round(ppix[1], 2)), (mosaic_window_width - 100, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        

    Ts = np.array([[pixels_per_meter, 0.0, 0.0],
                   [0.0, -pixels_per_meter, 0.0],   # flip Y here because image +Y is down but UTM +Y is up
                   [0.0, 0.0, 1.0]])
    Tt = np.array([[1.0, 0.0, -imagex[context_index]],
                   [0.0, 1.0, -imagey[context_index]], 
                   [0.0, 0.0, 1.0]])
#    T_xform = np.matmul(Ts, Tt)
    T_xform = Ts @ Tr @ Tt
    
#    ns_after = time.time_ns()
#    ns_diff = ns_after - ns_before
#    print("drawing rast images: ", ns_diff)

    utm_draw_color = (0, 150, 0)
    utm_draw_thickness = 2

    xshift = imxtrans+mosaic_window_width/2
    yshift = imytrans+mosaic_window_height/2
    xmin_xshift = 0 - xshift
    xmax_xshift = mosaic_window_width - 1 - xshift
    ymin_yshift = 0 - yshift
    ymax_yshift = mosaic_window_height - 1 - yshift
    
#    ns_before = time.time_ns()

    if imscale_factor >= 0.5:
        fontscale = 1.0
    else:
        fontscale = imscale_factor + 0.5

    # RANSAC lines here -- heading-aware

    xstart = imagex[leg_start_idx_list[leg_context_index]]
    ystart = imagey[leg_start_idx_list[leg_context_index]]
    xstart_proj, ystart_proj, _ = closest_point_on_line(leg_m_list[leg_context_index], leg_c_list[leg_context_index], xstart, ystart)

    xend = imagex[leg_end_idx_list[leg_context_index]]
    yend = imagey[leg_end_idx_list[leg_context_index]]
    xend_proj, yend_proj, _ = closest_point_on_line(leg_m_list[leg_context_index], leg_c_list[leg_context_index], xend, yend)

    pt_start = np.array([xstart, ystart, 1.0])
    p_start = np.transpose(pt_start)
    ppix_start = np.matmul(T_xform, p_start)
    
    pt_start_proj = np.array([xstart_proj, ystart_proj, 1.0])
    p_start_proj = np.transpose(pt_start_proj)
    ppix_start_proj = np.matmul(T_xform, p_start_proj)

    cv2.circle(draw_im, (int(ppix_start[0] + xshift), int(ppix_start[1] + yshift)), 5, (255, 0, 0), 2)
    cv2.circle(draw_im, (int(ppix_start_proj[0] + xshift), int(ppix_start_proj[1] + yshift)), 5, (0, 255, 0), 2)
    
    pt_end = np.array([xend, yend, 1.0])
    p_end = np.transpose(pt_end)
    ppix_end = np.matmul(T_xform, p_end)
    
    pt_end_proj = np.array([xend_proj, yend_proj, 1.0])
    p_end_proj = np.transpose(pt_end_proj)
    ppix_end_proj = np.matmul(T_xform, p_end_proj)

    cv2.circle(draw_im, (int(ppix_end[0] + xshift), int(ppix_end[1] + yshift)), 5, (255, 0, 0), 2)
    cv2.circle(draw_im, (int(ppix_end_proj[0] + xshift), int(ppix_end_proj[1] + yshift)), 5, (0, 0, 255), 2)
    
    cv2.line(draw_im, \
             (int(ppix_start_proj[0] + xshift), int(ppix_start_proj[1] + yshift)), \
             (int(ppix_end_proj[0] + xshift), int(ppix_end_proj[1] + yshift)), \
             (0, 100, 100), 1)

    # leg neighbors

    if image_prev_neighbor_idx[idx] != -1:
        pt = np.array([imagex[image_prev_neighbor_idx[idx]], imagey[image_prev_neighbor_idx[idx]], 1.0])
        p = np.transpose(pt)
        ppix = np.matmul(T_xform, p)
        if ppix[0] >= xmin_xshift and ppix[0] <= xmax_xshift and ppix[1] >= ymin_yshift and ppix[1] <= ymax_yshift:
            cv2.circle(draw_im, \
                       (int(ppix[0] + xshift), \
                        int(ppix[1] + yshift)), \
                       int(imscale_factor*100.0), (0, 0, 255), -1)

    if image_next_neighbor_idx[idx] != -1:
        pt = np.array([imagex[image_next_neighbor_idx[idx]], imagey[image_next_neighbor_idx[idx]], 1.0])
        p = np.transpose(pt)
        ppix = np.matmul(T_xform, p)
        if ppix[0] >= xmin_xshift and ppix[0] <= xmax_xshift and ppix[1] >= ymin_yshift and ppix[1] <= ymax_yshift:
            cv2.circle(draw_im, \
                       (int(ppix[0] + xshift), \
                        int(ppix[1] + yshift)), \
                       int(imscale_factor*100.0), (255, 0, 0), -1)

    
    # image centers by UTM
    
    for i in range(start_index, stop_index):
        
        if math.isnan(imagex[i]) or math.isnan(imagey[i]):
            continue

        pt = np.array([imagex[i], imagey[i], 1.0])
        p = np.transpose(pt)
        ppix = np.matmul(T_xform, p)

        #if False:
        if ppix[0] >= xmin_xshift and ppix[0] <= xmax_xshift and ppix[1] >= ymin_yshift and ppix[1] <= ymax_yshift:
            cv2.circle(draw_im, \
                       (int(ppix[0] + xshift), \
                        int(ppix[1] + yshift)), \
                       int(imscale_factor*8), utm_draw_color, utm_draw_thickness)
            if draw_frame_info:
                cv2.putText(draw_im, str(i), (int(5 + ppix[0] + xshift), int(5 + ppix[1] + yshift)), cv2.FONT_HERSHEY_PLAIN, fontscale, (255, 255, 255), 1)
                cv2.putText(draw_im, str(image_altitude[i]), (int(5 + ppix[0] + xshift), int(20 + ppix[1] + yshift)), cv2.FONT_HERSHEY_PLAIN, fontscale, (255, 255, 255), 1)

    # sonar waterfall images

#    ns_before = time.time_ns()

    if do_sonar:

#        print("sonar disc", len(sonar_discontinuity_flag))
#        print("sonar overlap", len(sonar_overlap_row))
        
        sonimcorns = np.zeros((4, 2))
        
        for i in range(len(image_in_sonarrect_idx_list)):
#        for j in range(numsonars):

            j = image_in_sonarrect_idx_list[i]
            print("!!!!!:",image_in_sonarrect_idx_list)
#            print(i, len(image_in_sonarrect_idx_list))
            
            if i == current_sonarrect_idx:
                sonar_draw_color = (255, 0, 255)
                sonar_draw_thickness = 2
            else:
                sonar_draw_color = (25, 0, 25)
                sonar_draw_thickness = 1

            # skip NaNs because corners are meaningless
            
            if math.isnan(sonarx[j]) or math.isnan(sonary[j]):
                continue
            
            # get sonar rectangle corners in the mosaic image
        
            sonar_T_rect2utm = np.array([[1.0, 0.0, sonarx_leg_proj[j] + sonar_alignment_factor_list[i] * sonardx[j]],    # sonarx
                                         [0.0, 1.0, sonary_leg_proj[j] + sonar_alignment_factor_list[i] * sonardy[j]],    # sonary
                                         [0.0, 0.0, 1.0]])

            sonar_TR_rect2utm = np.matmul(sonar_T_rect2utm, sonar_R_rect2utm_list[j])
            #print("TR:", sonar_TR_rect2utm)
            T_combined = np.matmul(T_xform, sonar_TR_rect2utm)
            image_sonarcorners = np.matmul(T_combined, sonarcorners)
            #print("corner: ", image_sonarcorners)
#            print(sonarx[j], sonary[j])

            # draw sonar rectangle

            for corner_idx in range(4):
                sonimcorns[corner_idx][0] = imxtrans+image_sonarcorners[0][corner_idx] + mosaic_window_width/2
                sonimcorns[corner_idx][1] = imytrans+image_sonarcorners[1][corner_idx] + mosaic_window_height/2
            
            for corner_idx in range(4):
                cv2.line(draw_im, \
                         (int(sonimcorns[corner_idx][0]), int(sonimcorns[corner_idx][1])),
                         (int(sonimcorns[(corner_idx+1)%4][0]), int(sonimcorns[(corner_idx+1)%4][1])),                     
                         sonar_draw_color, sonar_draw_thickness)
            

            # draw overlap line
            
            # 0 is "front" left, 1 is "front" right
            # 3 is "back" left, 2 is "back" right
            # 0->3 is parallel to 1->2
#            print("problem", j, len(sonar_overlap_row))
            # len(sonar_overlap_row) = numsonars - 1 because its PAIRS
            if j < numsonars - 1:
                t = float(sonar_overlap_row[j]) / float(sonar_im_height)
                xdiff = float(image_sonarcorners[0][3] - image_sonarcorners[0][0])
                ydiff = float(image_sonarcorners[1][3] - image_sonarcorners[1][0])

#            print("{0:d}, {1:d}, {2:.3f}, {3:.3f}, {4:.3f}, {5:d}".format(j, sonar_overlap_row[j], t, xdiff, ydiff, sonar_discontinuity_flag[j]))

                overlap_p0_x = float(sonimcorns[0][0]) + t * xdiff
                overlap_p0_y = float(sonimcorns[0][1]) + t * ydiff
                overlap_p1_x = float(sonimcorns[1][0]) + t * xdiff
                overlap_p1_y = float(sonimcorns[1][1]) + t * ydiff

                if sonar_discontinuity_flag[j]:
                    overlap_line_color = (0, 0, 255)
                else:
                    overlap_line_color = (255, 0, 0)
                
                cv2.line(draw_im, \
                         (int(overlap_p0_x), int(overlap_p0_y)),
                         (int(overlap_p1_x), int(overlap_p1_y)),
                         overlap_line_color, sonar_draw_thickness)


            # label rectangle with sonar index
        
            cv2.putText(draw_im, str(j), \
                        (int(sonimcorns[0][0]), int(sonimcorns[0][1] - 5)),
#                        cv2.FONT_HERSHEY_PLAIN, imscale_factor*8.0, sonar_draw_color, 1)
#                        cv2.FONT_HERSHEY_PLAIN, 1.0, sonar_draw_color, 1)
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

#    ns_after = time.time_ns()
#    ns_diff = ns_after - ns_before
#    print("drawing rast sonar: ", ns_diff)

#-------------------------------------------------------------------------------------

# this doesn't get called anymore

def draw_sonar_overlays(s_im, s_idx):
    cv2.putText(s_im, str(s_idx), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    cv2.putText(s_im, str(round(sonarx[s_idx], 2)) + " " + str(round(sonary[s_idx], 2)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    cv2.putText(s_im, str(round(sonardx[s_idx], 2)) + " " + str(round(sonardy[s_idx], 2)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

#-------------------------------------------------------------------------------------

# not closest per se, but sort of the "ceiling" closest -- the next one with index greater than cam_idx

def compute_sonar_context_index(cam_idx):

    min_diff = 1000000
    min_sonar_index = -1
    
    for j in range(numsonars):
        diff = cam_minindex[j] - cam_idx
        #print(cam_idx, j, diff, min_diff)
        if diff >= 0 and diff < min_diff:
            min_diff = diff
            min_sonar_index = j
            
    return min_sonar_index

#-------------------------------------------------------------------------------------

# call after legs have been computed

def compute_all_prev_and_next_leg_neighbors():

    leg_neighbors_filename = mission_name + "_leg_neighbors.csv"

    if path.exists(leg_neighbors_filename):
        print("reading leg neighbors!")

        with open(leg_neighbors_filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            image_idx = 0
            for row in spamreader:
                image_prev_neighbor_idx[image_idx] = row[2] 
                image_next_neighbor_idx[image_idx] = row[4] 
                image_idx += 1
    else:
        print("writing leg neighbors!")
        f = open(leg_neighbors_filename, "w")
        
        for image_idx in range(num_imageframes):
            leg_idx = compute_leg_idx_from_image_idx(image_idx)
            imagex_leg_proj[image_idx], imagey_leg_proj[image_idx], image_leg_proj_dist[image_idx] = \
                closest_point_on_line(leg_m_list[leg_idx], leg_c_list[leg_idx], imagex[image_idx], imagey[image_idx])

        for image_idx in range(num_imageframes):

            if not (image_idx % 100):
                print("compute all", image_idx, num_imageframes)
            
            leg_idx = compute_leg_idx_from_image_idx(image_idx)
            prev_leg_idx = leg_idx - 1
            next_leg_idx = leg_idx + 1

#            print("********************************* image", image_idx, leg_idx)
            # if prev_leg_idx is a real leg and not too far, set index.  else, don't do anything because -1 is already there
            min_prev_pt_dist = 999.0
            if prev_leg_idx >= 0:
                imagex_prev_leg_proj, imagey_prev_leg_proj, prev_leg_dist = \
                    closest_point_on_line(leg_m_list[prev_leg_idx], leg_c_list[prev_leg_idx], imagex[image_idx], imagey[image_idx])
                #xdiff = imagex_prev_leg_proj - imagex[image_idx]
                #ydiff = imagey_prev_leg_proj - imagey[image_idx]
                #prev_leg_dist = math.sqrt(xdiff*xdiff + ydiff*ydiff)
                # if prev leg not too far away, iterate through every point in it and find closest
                if prev_leg_dist <= max_leg_dist:
                    min_prev_pt_idx = -1
                    for idx in range(leg_start_idx_list[prev_leg_idx], leg_end_idx_list[prev_leg_idx] + 1):
                        xdiff = imagex_prev_leg_proj - imagex[idx]
                        ydiff = imagey_prev_leg_proj - imagey[idx]
                        dist = math.sqrt(xdiff*xdiff + ydiff*ydiff)
  #                      print("checking prev", idx, dist, min_prev_pt_dist)
                        if dist < min_prev_pt_dist:
                            min_prev_pt_dist = dist
                            min_prev_pt_idx = idx
                    if min_prev_pt_dist < max_leg_dist:
                        image_prev_neighbor_idx[image_idx] = min_prev_pt_idx
                        
#            print("prev winner", image_prev_neighbor_idx[image_idx])
            
            # if next_leg_idx is a real leg and not too far
            min_next_pt_dist = 999.0                
            if next_leg_idx < len(leg_start_idx_list):
                imagex_next_leg_proj, imagey_next_leg_proj, next_leg_dist = \
                    closest_point_on_line(leg_m_list[next_leg_idx], leg_c_list[next_leg_idx], imagex[image_idx], imagey[image_idx])
                #xdiff = imagex_next_leg_proj - imagex[image_idx]
                #ydiff = imagey_next_leg_proj - imagey[image_idx]
                #next_leg_dist = math.sqrt(xdiff*xdiff + ydiff*ydiff)
                # if next leg not too far, iterate through every point in it and find closest
                if next_leg_dist <= max_leg_dist:
                    min_next_pt_idx = -1
                    for idx in range(leg_start_idx_list[next_leg_idx], leg_end_idx_list[next_leg_idx] + 1):
                        xdiff = imagex_next_leg_proj - imagex[idx]
                        ydiff = imagey_next_leg_proj - imagey[idx]
                        dist = math.sqrt(xdiff*xdiff + ydiff*ydiff)
 #                       print("checking next", idx, dist, min_next_pt_dist)
                        if dist < min_next_pt_dist:
                            min_next_pt_dist = dist
                            min_next_pt_idx = idx
                    if min_next_pt_dist < max_leg_dist:
                        image_next_neighbor_idx[image_idx] = min_next_pt_idx

#            print("next winner", image_next_neighbor_idx[image_idx])
                
            f.write(str(image_idx) + " " + str(leg_idx) + " " + \
                    str(image_prev_neighbor_idx[image_idx]) + " " + str(round(min_prev_pt_dist, 3)) + " " + \
                    str(image_next_neighbor_idx[image_idx]) + " " + str(round(min_next_pt_dist, 3)) + "\n")
        f.close()
        sys.exit()

#-------------------------------------------------------------------------------------
def write_to_file(imagelist, x,y,dx,dy):
    with open('/home/jiayi/sonar/explore.csv','w+') as csvfile:
        for i in range(numsonars):
            newname = imagelist[i][:-3] + 'png'
            newline = ','.join(str(a) for a in [newname, x[i],y[i],dx[i],dy[i]])
            #print(newline)
            csvfile.write("%s\n" % newline)
#-------------------------------------------------------------------------------------
# preamble


M = loadmat(metadatafilename)
camera = M["camera"][0][0]
num_imageframes = camera['filename'].shape[0]
print(num_imageframes)
#sys.exit()

imagex = np.zeros(num_imageframes)
imagey = np.zeros(num_imageframes)
imagedx = np.zeros(num_imageframes)
imagedy = np.zeros(num_imageframes)

imagex_leg_proj = np.zeros(num_imageframes)
imagey_leg_proj = np.zeros(num_imageframes)
image_leg_proj_dist = np.zeros(num_imageframes)   # how far away is (imagex, imagey) from its leg projection?  can be used for outlier filtering

image_prev_neighbor_idx = np.full(num_imageframes, -1)
image_next_neighbor_idx = np.full(num_imageframes, -1)

image_farnext = np.zeros(num_imageframes)
image_altitude = np.zeros(num_imageframes)

compute_image_positions_and_headings()

if do_sonar:
#    generate_pair_list()
#    read_cache_sonar_images()
#    write_cache_sonar_images()
    preprocess_sonar()

all_nodes = []
all_nodes_set = set()
with open(neighbors_filename, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        all_nodes_set.add(int(row[0]))
        nodes = list(map(int, row[0:-1]))   # for some reason each list ends with '', so strip
#        if len(nodes) == 1:
#            print(nodes)
#            sys.exit()
        all_nodes.append(nodes)

# insert everything from 0 to num_imageframeselse as a no-neighbor node
for j in range(num_imageframes):
    if j not in all_nodes_set:
        all_nodes_set.add(j)
        all_nodes.append([j])
        
# then sort
all_nodes.sort(key=itemgetter(0))

start_index = all_nodes[0][0]
stop_index = all_nodes[-1][0]

#start_sonar_index = 0
#stop_sonar_index = numsonars

# default
context_index = start_index
#sonar_context_index = compute_sonar_context_index(context_index)
#sonar_index = start_sonar_index

if len(sys.argv) >= 2:
    index_val = int(sys.argv[1])
    if index_val < start_index:
        index_val = start_index
    elif index_val > stop_index:
        index_val = stop_index
    context_index = index_val

#-------------------------------------------------------------------------------------
#load_retinanet()
#-------------------------------------------------------------------------------------
# loop

DISPLAY_MODE_NORMAL  = 1
DISPLAY_MODE_MASK    = 2

new_context_image_flag = True
display_mode = DISPLAY_MODE_NORMAL
#display_mask_flag = False
#display_mosaic_flag = False
#display_pll_corners_flag = False

default_imxtrans = 0.0
default_imytrans = 0.0
imtrans_delta = 1.0
imxtrans = default_imxtrans
imytrans = default_imytrans

default_imscale_factor = 0.025
#default_imscale_factor = 0.5
imscale_factor = default_imscale_factor
default_imscale_delta = 0.001
#default_imscale_delta = 0.01
imscale_delta = default_imscale_delta
imscale_delta_switch = 0.1
pixels_per_meter = imscale_factor * imwidth / IMAGE_WIDTH_IN_METERS


# images are actually reverse sorted, so I'm just swapping the key binds
# imscale_factor does not affect context, just scales up/down the window and everything in it

#print("list: ",image_in_sonarrect_waterfall_im_list)
#print("write file")
#write_to_file(sonarlist, sonarx, sonary, sonardx, sonardy)

while True:

    # navigate to another image?
    
    if new_context_image_flag:
        imsuperdir, imsubdir_list, imname_list, idx_list, src_im_list, H_list, dst_im_scaled_list, dst_im_mask_scaled_list, ordered_indices, current_idx, imcenter_list, imcorners_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = initialize_context_image(context_index, imscale_factor)
        #print("imname: ", imname_list)
        #print("list: ", dst_im_scaled_list)
        new_context_image_flag = False
        H_utm2utm_list, H_im2utm_list = build_PLL_corners_list(current_idx, idx_list)
#        print(len(H_utm2utm_list))
#        print(len(H_im2utm_list))

    # whole mosaic

#    draw_sonar_overlays(sonar_im, sonar_index)
#    sonar_im = image_in_sonarrect_waterfall_im_list[current_sonarrect_idx]

    if do_sonar:
        if not dst_sonar_im_scaled_list:
            #print("no sonars!")
            current_sonar_exists = False
#            sys.exit()
        elif current_sonarrect_idx < 0 or current_sonarrect_idx >= len(dst_sonar_im_scaled_list):
            print(current_sonarrect_idx, len(dst_sonar_im_scaled_list))
            #print("current sonar index out of range")
            #sys.exit()
            current_sonar_exists = False
        else:
            current_sonar_exists = True

#    ns_before = time.time_ns()


# get retinanet rock bboxes



    if display_mode == DISPLAY_MODE_NORMAL:
        if do_sonar and current_sonar_exists:
            foreground = dst_im_scaled_list[ordered_indices[current_idx]]
            foreground = foreground.astype(float)/255.0
            #cv2.imshow("test", foreground)
            background = dst_sonar_im_scaled_list[current_sonarrect_idx]
            background = background.astype(float)/255.0
            alpha = cv2.cvtColor(dst_im_mask_scaled_list[ordered_indices[current_idx]], cv2.COLOR_GRAY2BGR)
            alpha = alpha.astype(float)/255.0
            if draw_camera_over_sonar:
                foreground = cv2.multiply(alpha, foreground)
                background = cv2.multiply(1.0 - alpha, background)
                draw_im_float = 255.0*cv2.add(foreground, background)
            else:
                draw_im_float = 255.0*background
        
            draw_im = draw_im_float.astype(np.uint8)
        else:
            draw_im = np.copy(dst_im_scaled_list[ordered_indices[current_idx]])
    elif display_mode == DISPLAY_MODE_MASK:
#        if do_sonar and current_sonar_exists:
#            foreground = cv2.cvtColor(dst_im_mask_scaled_list[ordered_indices[current_idx]], cv2.COLOR_GRAY2BGR)
#            foreground = foreground.astype(float)/255.0
#            background = dst_sonar_nadir_mask_im_scaled_list[current_sonarrect_idx]
#            background = background.astype(float)/255.0
#            alpha = cv2.cvtColor(dst_im_mask_scaled_list[ordered_indices[current_idx]], cv2.COLOR_GRAY2BGR)
#            alpha = alpha.astype(float)/255.0
#            if draw_camera_over_sonar:
#                foreground = cv2.multiply(alpha, foreground)
#                background = cv2.multiply(1.0 - alpha, background)
#                draw_im_float = 255.0*cv2.add(foreground, background)
#            else:
#                draw_im_float = 255.0*background
#            
#            draw_im = draw_im_float.astype(np.uint8)
#        else:
#            draw_im = np.copy(cv2.cvtColor(dst_im_mask_scaled_list[ordered_indices[current_idx]], cv2.COLOR_GRAY2BGR))
        draw_im = np.zeros((mosaic_window_height, mosaic_window_width, 3), np.uint8)

#    ns_after = time.time_ns()
#    ns_diff = ns_after - ns_before
#    print("drawing rasterization: ", ns_diff)

    # overlay

#    ns_before = time.time_ns()
    #cv2.imshow("test", draw_im)
    draw_normal_overlays(draw_im, imsubdir_list, imname_list, ordered_indices, current_idx, imcenter_list, imcorners_list, idx_list, image_in_sonarrect_idx_list, current_sonarrect_idx)

#    ns_after = time.time_ns()
#    ns_diff = ns_after - ns_before
#    print("drawing overlays: ", ns_diff)

    # show it
    #rock_im = load_sonar_image_with_rock(current_sonarrect_idx)

    #print("key:", current_sonarrect_idx)


    cv2.imshow(str(imsuperdir)[:-1], draw_im)

    #cv2.imshow("retinanet", rock_im)

    cv2.imshow("image", src_im_list[ordered_indices[current_idx]])
    #print(str(imsuperdir)[:-1])
#    cv2.imshow("sonar", sonar_im)
#    cv2.imshow("dst sonar", dst_sonar_im_scaled_list[current_sonarrect_idx])
#    cv2.imshow("right", sonar_right_im)
    key = cv2.waitKey(0)

    # handle keypresses

    #write_to_file(sonarlist, sonarx, sonary, sonardx, sonardy)
    
    # flip order of camera and sonar in overlay

    if key==ord(' '):
        draw_camera_over_sonar = not draw_camera_over_sonar

    # reset scale and origin of image mosaic

    if key==ord('0'):
        imxtrans = default_imxtrans
        imytrans = default_imytrans
        imscale_factor = default_imscale_factor
        imscale_delta = default_imscale_delta
        pixels_per_meter = imscale_factor * imwidth / IMAGE_WIDTH_IN_METERS
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    # translate image mosaic

    if key==ord('d'):
        imxtrans += imtrans_delta
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)
        
    if key==ord('a'):
        imxtrans -= imtrans_delta
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    if key==ord('w'):
        imytrans -= imtrans_delta
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

        
    if key==ord('s'):
        imytrans += imtrans_delta
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    if key==ord('h'):
        imxtrans += 10.0*imtrans_delta
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)
        
    if key==ord('f'):
        imxtrans -= 10.0*imtrans_delta
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    if key==ord('t'):
        imytrans -= 10.0*imtrans_delta
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    if key==ord('g'):
        imytrans += 10.0*imtrans_delta
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    # go to prev leg neighbor
    if key==ord('k') and image_prev_neighbor_idx[context_index] != -1:
        context_index = image_prev_neighbor_idx[context_index]
        new_context_image_flag = True
    
    # go to next leg neighbor
    if key==ord('l') and image_next_neighbor_idx[context_index] != -1:
        context_index = image_next_neighbor_idx[context_index]
        new_context_image_flag = True
    
    # hop to preselected zoom level 

    if key==ord('5'):
        imscale_factor = default_imscale_factor/30.0
        pixels_per_meter = imscale_factor * imwidth / IMAGE_WIDTH_IN_METERS
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    # hop to preselected zoom level 

    if key==ord('6'):
        imscale_factor = default_imscale_factor/10.0
        pixels_per_meter = imscale_factor * imwidth / IMAGE_WIDTH_IN_METERS
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    # hop to preselected zoom level 

    if key==ord('7'):
        imscale_factor = default_imscale_factor/3.0
        pixels_per_meter = imscale_factor * imwidth / IMAGE_WIDTH_IN_METERS
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    # hop to preselected zoom level 

    if key==ord('8'):
        imscale_factor = 10.0*default_imscale_factor
        pixels_per_meter = imscale_factor * imwidth / IMAGE_WIDTH_IN_METERS
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    # hop to preselected zoom level 

    if key==ord('9'):
        imscale_factor = 20.0*default_imscale_factor
        pixels_per_meter = imscale_factor * imwidth / IMAGE_WIDTH_IN_METERS
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    # magnify image mosaic

    if key==ord('=') and imscale_factor < 2.0:
        #print(imscale_factor)
        if imscale_factor <= imscale_delta_switch and imscale_factor + imscale_delta >= imscale_delta_switch:
            #print("scaling up")
            imscale_factor += imscale_delta            
            imscale_delta *= 10.0
        imscale_factor += imscale_delta            
        pixels_per_meter = imscale_factor * imwidth / IMAGE_WIDTH_IN_METERS
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)
        
    # minify image mosaic
    
    if key==ord('-') and imscale_factor > imscale_delta:
        #print(imscale_factor)
        if imscale_factor >= imscale_delta_switch and imscale_factor - imscale_delta <= imscale_delta_switch:
            #print("scaling down")
            imscale_factor -= imscale_delta
            imscale_delta /= 10.0
        imscale_factor -= imscale_delta
        pixels_per_meter = imscale_factor * imwidth / IMAGE_WIDTH_IN_METERS
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)
    
    # next neighbor camera image
    if key==ord('\''):
        current_idx -= 1
        if current_idx < 0:
            current_idx = len(dst_im_scaled_list)-1

    # previous neighbor camera image
    elif key==ord(';'):
        current_idx += 1
        if current_idx >= len(dst_im_scaled_list):
            current_idx = 0

    # previous sonar
    elif key== ord('z'):
        num_sonarrects = len(image_in_sonarrect_idx_list)
        #print(current_sonarrect_idx, "/", num_sonarrects)
        if current_sonarrect_idx > 0:
            current_sonarrect_idx -= 1
        else:
            current_sonarrect_idx = num_sonarrects - 1

    elif key == ord('x'):
        print(current_sonarrect_idx)
        num_sonarrects = len(image_in_sonarrect_idx_list)
        #print(current_sonarrect_idx, "/", num_sonarrects)
        if current_sonarrect_idx < num_sonarrects - 1:
            current_sonarrect_idx += 1
        else:
            current_sonarrect_idx = 0

    # next absolute image (changes neighbors)
    elif key==ord(']'):
        dummysuperdir, dummysubdir, dummyname, file_exists = get_image_filename(camera['filename'][context_index + 1][0][0])
        if context_index + 1 <= stop_index:
            context_index += 1
            new_context_image_flag = True
        else:
            print("cannot increment -- out of range")

    # previous absolute image (changes neighbors)
    elif key==ord('['):
        dummysuperdir, dummysubdir, dummyname, file_exists = get_image_filename(camera['filename'][context_index - 1][0][0])
        if context_index - 1 >= start_index:
            context_index -= 1
            new_context_image_flag = True
        else:
            print("cannot decrement -- no image there")

    # jump forward 10
    elif key==ord('p'):
        dummysuperdir, dummysubdir, dummyname, file_exists = get_image_filename(camera['filename'][context_index + 10][0][0])
        if context_index + 10 <= stop_index:
            context_index += 10
            new_context_image_flag = True
        else:
            print("cannot increment 10 -- out of range")

    # jump backward 10
    elif key==ord('o'):
        dummysuperdir, dummysubdir, dummyname, file_exists = get_image_filename(camera['filename'][context_index - 10][0][0])
        if context_index - 10 >= start_index:
            context_index -= 10
            new_context_image_flag = True
        else:
            print("cannot decrement 10 -- no image there")

    # jump forward 100
    elif key==ord('i'):
        dummysuperdir, dummysubdir, dummyname, file_exists = get_image_filename(camera['filename'][context_index + 100][0][0])
        if context_index + 100 <= stop_index:
            context_index += 100
            new_context_image_flag = True
        else:
            print("cannot increment 100 -- out of range")

    # jump backward 100
    elif key==ord('u'):
        dummysuperdir, dummysubdir, dummyname, file_exists = get_image_filename(camera['filename'][context_index - 100][0][0])
        if context_index - 100 >= start_index:
            context_index -= 100
            new_context_image_flag = True
        else:
            print("cannot decrement 100 -- no image there")

    # toggle display of frame indices and altitudes
    elif key==ord('m'):
        draw_frame_info = not draw_frame_info

    elif key==ord('b'):
        sonar_alignment_factor -= 0.1
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)
    elif key==ord('n'):
        sonar_alignment_factor += 0.1
        dst_im_scaled_list, dst_im_mask_scaled_list, dst_sonar_im_scaled_list, dst_sonar_nadir_mask_im_scaled_list = refresh_transformed_images(imscale_factor, src_im_list, H_list, image_in_sonarrect_idx_list, image_in_sonarrect_waterfall_im_list, image_in_sonarrect_waterfall_nadir_mask_im_list)

    # change display mode
    elif key==ord('1'):
        display_mode = DISPLAY_MODE_NORMAL
    elif key==ord('2'):
        display_mode = DISPLAY_MODE_MASK

    # exit
    elif key & 0xFF ==ord('q'):
        sys.exit()
            
    



