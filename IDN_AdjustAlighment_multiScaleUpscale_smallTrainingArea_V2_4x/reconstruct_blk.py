import matplotlib
matplotlib.use("Pdf")

import os,sys
import caffe

import numpy as np
from PIL import Image

from skimage.transform import resize

import scipy.io

import cv2

downsample_factor = 4.0

model_file = './IDN_deploy_x4.prototxt'
weight_file = './snapshot_caffemodel_x4/IDN_iter_330000.caffemodel'

#im_file_gt = '/home/yanghu/dataset/GeoDataLondon/DataMoreAreas/Remodelling_Area/P5M/LIDAR_Remodeling_Area1_p5m.mat'
#im_file_lr = '/home/yanghu/dataset/GeoDataLondon/DataMoreAreas/Remodelling_Area/2M/LIDAR_Remodeling_Area1_2m.mat'

#im_file_gt = '/home/yanghu/dataset/GeoDataLondon/DataMoreAreas/Remodelling_Area/P5M/LIDAR_Remodeling_Area2_p5m.mat'
#im_file_lr = '/home/yanghu/dataset/GeoDataLondon/DataMoreAreas/Remodelling_Area/2M/LIDAR_Remodeling_Area2_2m.mat'

#im_file_gt = '/home/yanghu/dataset/GeoDataLondon/DataMoreAreas/Remodelling_Area/P5M/LIDAR_Remodeling_Area3_p5m.mat'
#im_file_lr = '/home/yanghu/dataset/GeoDataLondon/DataMoreAreas/Remodelling_Area/2M/LIDAR_Remodeling_Area3_2m.mat'

blk_size = 250
blk_overlap = 125

blk_size_up = int(downsample_factor*blk_size)

st_crop = int(downsample_factor-1)
st_gt_shift = 2
blk_size_up_crop = int(blk_size_up - downsample_factor + 1)

#initialize network
caffe.set_device(3)
caffe.set_mode_gpu()

net = caffe.Net(model_file, weight_file, caffe.TEST)

#im_lr = cv2.imread(im_file_lr, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

im_lr = scipy.io.loadmat(im_file_lr)['im']
im_gt = scipy.io.loadmat(im_file_gt)['im']

m_resize = im_lr.shape[0]
n_resize = im_lr.shape[1]

m = int(downsample_factor*m_resize)
n = int(downsample_factor*n_resize)

im_lr_up = cv2.resize(im_lr,(n,m),interpolation = cv2.INTER_NEAREST)

imRst = np.zeros((m,n),dtype = np.float32)
imRstWeight = np.zeros((m,n),dtype = np.float32)
imRstFinal = np.zeros((m,n),dtype = np.float32)

#m = int(downsample_factor * blk_size)
#n = int(downsample_factor * blk_size)

#divide lr image into overlapping blocks and reconstruct each block
xRange = np.arange(0, n_resize - blk_size + 1, blk_size-blk_overlap)
yRange = np.arange(0, m_resize - blk_size + 1, blk_size-blk_overlap)

#print m_resize
#print n_resize

print xRange
print yRange

mx,my = np.meshgrid(xRange,yRange)

mx = mx.flatten()
my = my.flatten()

#print mx
#print my

for i in xrange(0,len(mx)):
    #get image blocks
    tmpX = mx[i]
    tmpY = my[i]
    tmpImLR = im_lr[tmpY:tmpY+blk_size,tmpX:tmpX+blk_size]
    tmpImLR[np.where(tmpImLR<-3e+38)]=0

    tmpImLR_up = cv2.resize(tmpImLR,(blk_size_up,blk_size_up),interpolation = cv2.INTER_NEAREST)
    tmpImLR_up_crop = tmpImLR_up[st_crop:st_crop+blk_size_up_crop,st_crop:st_crop+blk_size_up_crop]

    #print tmpImLR.shape
    #print tmpImLR_up_crop.shape

    #reconstruction
    net.blobs['img'].data[0,0] = np.array(tmpImLR, dtype=np.float32)

    net.forward()

    tmpRst = np.array(net.blobs['sum_4x'].data[0,0],dtype=np.float32)
    tmpRst = tmpRst[:-st_gt_shift,:-st_gt_shift]

    x1 = int(downsample_factor*tmpX) + st_crop + st_gt_shift
    x2 = x1 + tmpRst.shape[1] 

    y1 = int(downsample_factor*tmpY) + st_crop + st_gt_shift
    y2 = y1 + tmpRst.shape[0]

    imRst[y1:y2,x1:x2] = imRst[y1:y2,x1:x2]+tmpRst    
    imRstWeight[y1:y2,x1:x2] = imRstWeight[y1:y2,x1:x2]+1
    #put the results back  

    #break

imRst[np.where(imRstWeight==0)]=-3.4028e+38
imRstWeight[np.where(imRstWeight==0)]=1

imRstFinal=np.divide(imRst,imRstWeight)
imRstFinal[np.where(im_lr_up<-3e+38)]=-3.4028e+38

rst = dict()
#st['imLR_up']=im_lr_up
rst['imGt']=im_gt
rst['imRstFinal']=imRstFinal
#rst['imRst']=imRst
#rst['imRstWeight']=imRstWeight
#scipy.io.savemat('tmpRst/LIDAR_Remodeling_Area1_2m_to_p5m.mat',rst)
#scipy.io.savemat('tmpRst/LIDAR_Remodeling_Area2_2m_to_p5m.mat',rst)
#scipy.io.savemat('tmpRst/LIDAR_Remodeling_Area3_2m_to_p5m.mat',rst)

