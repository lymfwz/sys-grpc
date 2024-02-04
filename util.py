import copy
import random
import numpy as np
#import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import rotate
from scipy.ndimage import measurements
import torch
from scipy.ndimage import distance_transform_edt as distance
import os
import cv2

def one_hot(batch_label, num):
    result = np.eye(num)[batch_label]
    return result

def fit_cube_param(vol_dim, cube_size, ita):
    dim = np.asarray(vol_dim)
    # cube number and overlap along 3 dimensions
    fold = dim / cube_size + ita
    ovlap = np.ceil(np.true_divide((fold * cube_size - dim), (fold - 1)))
    ovlap = ovlap.astype('int')

    fold = np.ceil(np.true_divide((dim + (fold - 1) * ovlap), cube_size))
    fold = fold.astype('int')

    return fold, ovlap

def decompose_vol2cube(vol_data, batch_size, cube_size, ita):
    # (h,w,3)
    ori_data = vol_data
    cube_list = []
    # (h,w)
    vol_data = vol_data[:,:,0]
    fold, ovlap = fit_cube_param(vol_data.shape, cube_size, ita)
    dim = np.asarray(vol_data.shape)
    print(fold)
    # decompose
    for R in range(0, fold[0]):
        r_s = R * cube_size[0] - R * ovlap[0]
        r_e = r_s + cube_size[0]
        if r_e >= dim[0]:
            r_s = dim[0] - cube_size[0]
            r_e = r_s + cube_size[0]
        for C in range(0, fold[1]):
            c_s = C * cube_size[1] - C * ovlap[1]
            c_e = c_s + cube_size[1]
            if c_e >= dim[1]:
                c_s = dim[1] - cube_size[1]
                c_e = c_s + cube_size[1]
                # partition multiple channels
                # (256,256,3)
            cube_temp = ori_data[r_s:r_e, c_s:c_e, :]
            cube_batch = np.zeros([batch_size, cube_size[0], cube_size[1],3]).astype('float32')
            cube_batch[:,] = copy.deepcopy(cube_temp)
            # save
            cube_list.append(cube_batch)

    return cube_list
    
def compose_label_cube2vol(cube_list, vol_dim, cube_size, ita):
    # vol_dim = £¨h,w£© cube_size(256,256)
    # get parameters for compose
    fold, ovlap = fit_cube_param(vol_dim, cube_size, ita)
    # create label volume for all classes
    label_classes_mat = (np.zeros([vol_dim[0], vol_dim[1]])).astype('float32')
    idx_classes_mat = (np.zeros([cube_size[0], cube_size[1]])).astype('float32')
    
    overlapping = (np.zeros([vol_dim[0], vol_dim[1]])).astype('float32')
    print(fold)
    p_count = 0
    for R in range(0, fold[0]):
        r_s = R * cube_size[0] - R * ovlap[0]
        r_e = r_s + cube_size[0]
        if r_e >= vol_dim[0]:
            r_s = vol_dim[0] - cube_size[0]
            r_e = r_s + cube_size[0]
        for C in range(0, fold[1]):
            c_s = C * cube_size[1] - C * ovlap[1]
            c_e = c_s + cube_size[1]
            if c_e >= vol_dim[1]:
                c_s = vol_dim[1] - cube_size[1]
                c_e = c_s + cube_size[1]
                # accumulation
            label_classes_mat[r_s:r_e, c_s:c_e] = label_classes_mat[r_s:r_e, c_s:c_e] + cube_list[p_count][0][0]
            if r_s==3413 and r_e==3669 and c_s==2885 and c_e==3141:
                image = cube_list[p_count][0][0]
                image[image>0.5]=255
                image[image<0.5]=0
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            # print(r_s,r_e,c_s,c_e)
            overlapping[r_s:r_e, c_s:c_e] = overlapping[r_s:r_e, c_s:c_e]+1
            p_count += 1
    print(label_classes_mat[-1,3090:3100])
    print(overlapping[-1,3090:3100])
    print(np.max(overlapping),np.min(overlapping))
    result = label_classes_mat/overlapping
    print(result[-1,3090:3100])
    result[result>0.5]=255
    result[result<0.5]=0

    return result
    
    