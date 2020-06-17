import matplotlib
matplotlib.use("Pdf")

import os,sys, argparse
import caffe

import numpy as np
from PIL import Image

from skimage.transform import resize

import scipy.io

import cv2

downsample_factor = 4.0

model_file = './IDN_deploy_x4.prototxt'
weight_file = './snapshot_caffemodel_x4/IDN_iter_340000.caffemodel'

im_file_gt = '/home/coyh4/dataset/GeoData/London/Remodeling_Area/Remodeling_Area2/Remodeling_Area2.tif'
im_file_lr = '/home/coyh4/dataset/GeoData/London/Remodeling_Area/Remodeling_Area2/Remodeling_Area2_2m.tif'

blk_size=1000

im_gt = cv2.imread(im_file_gt, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

im_gt = im_gt[10000:10000+blk_size,10000:10000+blk_size]
#im_gt = im_gt[16000:16000+blk_size,18000:18000+blk_size]

#st_crop = int(round((downsample_factor-1)/2.0))
st_crop = int(downsample_factor-1)
patch_size_crop = int(blk_size - downsample_factor + 1)
im_gt_crop = im_gt[st_crop:st_crop + patch_size_crop,st_crop:st_crop+ patch_size_crop]

'''
tmpIm = np.zeros((64,1,1000,1000))
tmpIm[0,:,:,:]=im_gt
tmpIm1=tmpIm[0]
print tmpIm1.shape
print im_gt[0:5,0:5]
print '****'
print tmpIm1[0,0:5,0:5]
'''

im_lr = cv2.imread(im_file_lr, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

im_lr = im_lr[2500:2500+int(blk_size/downsample_factor),2500:2500+int(blk_size/downsample_factor)]
#im_lr = im_lr[4000:4000+int(blk_size/downsample_factor),4500:4500+int(blk_size/downsample_factor)]

im_lr_up = cv2.resize(im_lr,(blk_size,blk_size),interpolation = cv2.INTER_NEAREST)
im_lr_up_crop = im_lr_up[st_crop:st_crop + patch_size_crop,st_crop:st_crop+ patch_size_crop]

caffe.set_device(0)
#caffe.set_mode_gpu()
caffe.set_mode_cpu()

net = caffe.Net(model_file, weight_file, caffe.TEST)

net.blobs['img'].data[0,0] = np.array(im_lr, dtype=np.float32)

net.forward()

print net.blobs['sum_4x'].data.shape

rst = np.array(net.blobs['sum_4x'].data[0,0],dtype=np.float32)

print np.mean(np.abs(rst-im_gt_crop))
print np.mean(np.abs(im_lr_up_crop-im_gt_crop))

tmpSaveRst = dict()
tmpSaveRst['img_HR']=rst
tmpSaveRst['img_LR_up']=im_lr_up_crop
tmpSaveRst['img_gt']=im_gt_crop

scipy.io.savemat('./tmpRst/tmpRst.mat',tmpSaveRst)
