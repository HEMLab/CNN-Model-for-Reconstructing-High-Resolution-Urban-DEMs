from __future__ import division
import caffe
import numpy as np
import os
import sys
import scipy.io
from datetime import datetime
from PIL import Image

def compute_loss(net, save_dir, test_iter, layers=['loss_2x','loss_4x','loss_8x','loss_16x']):
    loss = np.zeros(len(layers))
    for cnt in range(test_iter):
        #print 'test iter ' + str(cnt) + ' begin'
        net.forward()
        for i,name in enumerate(layers):
            loss[i] += net.blobs[name].data
        #for ls in loss:
            #print ls / float(cnt+1)

    if save_dir is not None:
        rst=dict()
        rst['im_downsample']=net.blobs['img'].data
        rst['im_reconstruct']=net.blobs['sum_8x'].data
        rst['im_gt']=net.blobs['img_gt_8x'].data
        scipy.io.savemat(os.path.join(save_dir,str(cnt)+'.mat'),rst)
    return loss / float(test_iter)

def sr_tests(solver, test_iter, layers=['loss_2x','loss_4x','loss_8x','loss_16x'], saveRstPath=None, saveFilePath=None):
    print '>>>', datetime.now(), 'Begin tests'
    solver.test_nets[0].share_with(solver.net)
    #solver.test_nets[0].copy_from(saveFilePath[:-15]+'train_iter_'+str(solver.iter)+'.caffemodel')
    do_sr_tests(solver.test_nets[0], solver.iter, test_iter, layers, saveRstPath, saveFilePath)

def do_sr_tests(net, solver_iter, test_iter, layers=['loss_2x','loss_4x','loss_8x','loss_16x'], saveRstPath=None, saveFilePath=None):
    loss = compute_loss(net, saveRstPath, test_iter, layers)
    for i,name in enumerate(layers):
        print '>>>', datetime.now(), 'Iteration', solver_iter, name, loss[i]

    if saveFilePath is not None:
        with open(saveFilePath,'ab+') as f:
            for i,name in enumerate(layers):
                tmpStr= '>>> ' + datetime.now().strftime("%Y/%m/%d %X ") + 'Iteration ' + str(solver_iter) + ' ' + name +' ' + str(loss[i]) + os.linesep
                f.write(tmpStr)
        f.close()
    return loss
