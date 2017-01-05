#!/usr/bin/env sh
set -e

./build/tools/caffe train -solver ../scripts/face_train_solver.prototxt -weights ../models/center/center_step2_addacc_iter_1500.caffemodel  -gpu 0 $@

# ../models/center/center_step2_iter_30000.caffemodel

# ../models/center/center_fixed_iter_3000.caffemodel
# ../models/center/center_step2_acc_iter_6000.caffemodel
# ../models/center/center_fixed_iter_3000.caffemodel