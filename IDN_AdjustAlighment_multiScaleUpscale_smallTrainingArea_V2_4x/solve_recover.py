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

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.AdamSolver('./IDN_solver.prototxt')

test_iter = 3000 #8x, patch 256x256 -> 1500; 4x, patch 128x128 -> 3000

solver.restore('./snapshot_caffemodel_x4/IDN_iter_290000.solverstate')

for _ in range(2000):
    solver.step(10000)
    score.sr_tests(solver,test_iter,layers=['loss_2x','loss_4x'],saveFilePath='./snapshot_caffemodel_x4/snapshot.txt')
