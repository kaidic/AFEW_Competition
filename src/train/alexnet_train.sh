#!/usr/bin/env sh
set -e

./build/tools/caffe train -solver ../scripts/alexnet_train_solver.prototxt -weights ../pretrained/alexnet/Submission_3.caffemodel  -gpu 2 $@

# ../models/center/center_step2_iter_30000.caffemodel

# ../models/center/center_fixed_iter_3000.caffemodel
# ../models/center/center_step2_acc_iter_6000.caffemodel
# ../models/center/center_fixed_iter_3000.caffemodel