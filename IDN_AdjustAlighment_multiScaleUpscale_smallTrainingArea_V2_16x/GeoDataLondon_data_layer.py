
import caffe
import os

import numpy as np
from PIL import Image
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt

import random

from skimage.transform import resize

import cv2

class GeoDataLondonDataLayer(caffe.Layer):
    """
    Load (input image, rotated input image, rotation label) from Microsoft Coco dataset
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters (update this):

        - data_dir: path to data dir
        - tops: list of tops to output from {image, label}
        - mean: tuple of mean values to subtract
        - scale: scaling factor to multiply
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        - batch_size: input batch size
        - debug: debug flag
        - debug_dir: path to debug output dir 

        """
        # config
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.tops = params['tops']
        self.scale = params.get('scale', 1.0)
        self.random = params.get('randomize', True)
        self.split = params['split']
        self.seed = params.get('seed', None)
        self.initialScale = params.get('initial_scale', 1)
        self.batch_size = params.get('batch_size', 16)
        self.patch_size = params.get('patch_size', 128)
        self.downsample_factor = params.get('downsample_factor', 4)
        self.debug = params.get('debug', False)
        self.debug_dir = params.get('debug_dir', './debug_output')
	self.cntSmp=0 #count the number of samples used for training

        # store top data for reshape + forward
        self.data = {}
        self.size_lr = self.patch_size/self.downsample_factor
	self.data['image']=np.zeros((self.batch_size,1,self.size_lr,self.size_lr))

        #upscaled LR image and ground truth image
        self.size_2x = 2*self.size_lr-1
        self.data['image_gt_2x']=np.zeros((self.batch_size,1,self.size_2x,self.size_2x))

        self.size_4x = 2*self.size_2x-1
        self.data['image_gt_4x']=np.zeros((self.batch_size,1,self.size_4x,self.size_4x))

        self.size_8x = 2*self.size_4x-1
        self.data['image_gt_8x']=np.zeros((self.batch_size,1,self.size_8x,self.size_8x))

        self.size_16x = 2*self.size_8x-1
        self.data['image_gt_16x']=np.zeros((self.batch_size,1,self.size_16x,self.size_16x))
       
        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/data_list.txt'.format(self.data_dir)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = [0]*self.batch_size
	
	self.totalSmp=len(self.indices)
	print 'total sample: '+str(self.totalSmp)

        # randomization: seed and pick
        random.seed(self.seed)
	for cnt_batch_idx in range(0,self.batch_size):
	    self.idx[cnt_batch_idx] = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit
        for cnt in range(self.batch_size):
            #print cnt
            self.data['image'][cnt,0,:,:],self.data['image_gt_2x'][cnt,0,:,:],self.data['image_gt_4x'][cnt,0,:,:],self.data['image_gt_8x'][cnt,0,:,:],self.data['image_gt_16x'][cnt,0,:,:] = self.load_data(self.indices[self.idx[cnt]])
        top[0].reshape(*self.data['image'].shape)
        top[1].reshape(*self.data['image_gt_2x'].shape)
        top[2].reshape(*self.data['image_gt_4x'].shape)
        top[3].reshape(*self.data['image_gt_8x'].shape)
        top[4].reshape(*self.data['image_gt_16x'].shape)

    def forward(self, bottom, top):
        #print 'forward'
        # assign output
        top[0].data[...] = self.data['image']
        top[1].data[...] = self.data['image_gt_2x']
        top[2].data[...] = self.data['image_gt_4x']
        top[3].data[...] = self.data['image_gt_8x']
        top[4].data[...] = self.data['image_gt_16x']

        #print top[1].data

        # pick next input
	for cnt_batch_idx in range(0,self.batch_size):
            self.idx[cnt_batch_idx] = random.randint(0, len(self.indices)-1)

	self.cntSmp = self.cntSmp+self.batch_size
        #shuffle the data when an epoch ends
        if self.cntSmp >= self.totalSmp:
	    self.cntSmp=0
            self.seed=random.randint(0, 5000)
            random.seed(self.seed)

    def backward(self, top, propagate_down, bottom):
        pass
    
    def load_data(self, idx):
        """
        Load input images,rotation angle label; preprocess input images for Caffe:
        - load an image
        - apply an initial scaling if set to
        - stack the original image and rotated image
        - preprocessing the input images
        """
        tmpImName=idx
        #print 'read data'+tmpImName
        #load image
        im = np.array(scipy.io.loadmat('{}/{}.mat'.format(self.data_dir, tmpImName))['imBlk'],dtype=np.float32)

        im_crop = self.random_crop_flip(im)
        im_crop_lr = cv2.resize(im_crop,(self.size_lr,self.size_lr),interpolation = cv2.INTER_NEAREST)

        im_crop_2x = cv2.resize(im_crop,(2*self.size_lr,2*self.size_lr),interpolation = cv2.INTER_NEAREST)
        im_crop_4x = cv2.resize(im_crop,(4*self.size_lr,4*self.size_lr),interpolation = cv2.INTER_NEAREST)
        im_crop_8x = cv2.resize(im_crop,(8*self.size_lr,8*self.size_lr),interpolation = cv2.INTER_NEAREST)
        im_crop_16x = np.array(im_crop)

        if self.debug:
            if not os.path.isdir(self.debug_dir):
	        os.mkdir(self.debug_dir)
            rst=dict()
            rst['im_crop_lr']=im_crop_lr
            rst['im_crop_2x']=im_crop_2x[-self.size_2x:,-self.size_2x:]
            rst['im_crop_4x']=im_crop_4x[-self.size_4x:,-self.size_4x:]
            rst['im_crop_8x']=im_crop_8x[-self.size_8x:,-self.size_8x:]
            rst['im_crop_16x']=im_crop_16x[-self.size_16x:,-self.size_16x:]
            scipy.io.savemat(os.path.join(self.debug_dir,tmpImName),rst)

        return im_crop_lr,im_crop_2x[-self.size_2x:,-self.size_2x:],im_crop_4x[-self.size_4x:,-self.size_4x:],im_crop_8x[-self.size_8x:,-self.size_8x:],im_crop_16x[-self.size_16x:,-self.size_16x:] #crop HR resolution to match the output size

    def random_crop_flip(self,im):
        '''
        logics to apply crop and random flip to the input image
        '''
         
        tmpIm = np.array(im)
        isValid = False
        cnt_try = 0
        while (not isValid) and (cnt_try<100):
            ptX = random.randint(0, tmpIm.shape[1]-self.patch_size)
            ptY = random.randint(0, tmpIm.shape[0]-self.patch_size)
            rst = tmpIm[ptY:ptY+self.patch_size,ptX:ptX+self.patch_size]
            cnt_try = cnt_try + 1
            if len(np.where(rst<-3e+38)[0])==0:
                isValid=True
        if random.random() > 0.5:
            rst = np.fliplr(rst)
        rst[np.where(rst<-3e+38)]=0
        return rst
        
    def pre_process(self,im):
        '''
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - scale by scaling factor
        - transpose to channel x height x width order
        '''
        '''
        rst= np.array(im, dtype=np.float32)
        rst = rst[:,:,::-1]
        rst -= self.mean
        rst = rst * self.scale;
        rst = rst.transpose((2,0,1))
        return rst
        '''
