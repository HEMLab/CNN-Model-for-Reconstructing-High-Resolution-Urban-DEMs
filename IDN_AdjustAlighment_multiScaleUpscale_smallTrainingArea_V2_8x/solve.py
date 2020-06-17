import matplotlib
matplotlib.use("Pdf")

import os
import sys

import caffe
import score

import numpy as np

import surgery

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

caffe.set_device(1)
caffe.set_mode_gpu()

solver = caffe.AdamSolver('./IDN_solver.prototxt')

test_iter = 2000 #8x, patch 256x256 -> 1500; 4x, patch 128x128 -> 3000

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'img_up' in k]
factors = [2,2,2]
surgery.interp_NN(solver.net, interp_layers, factors)

#print interp_layers

#for l in interp_layers:
    #print solver.net.params[l][0].data
    #print solver.net.params[l][0].data.shape

#solver.step(10)

#for l in interp_layers:
    #print solver.net.params[l][0].data
    #print solver.net.params[l][0].data.shape

for _ in range(2000):
    solver.step(10000)
    score.sr_tests(solver,test_iter,saveFilePath='./snapshot_caffemodel_x8/snapshot.txt')
