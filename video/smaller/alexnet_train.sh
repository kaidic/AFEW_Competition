#!/usr/bin/env sh
set -e

./build/tools/caffe train -solver ../scripts/smaller/alexnet_train_smaller_solver.prototxt -weights ../models/smaller/alexnet_train_smaller_fer_iter_1000.caffemodel  -gpu 2 $@

# ../models/center/center_step2_iter_30000.caffemodel

# ../models/center/center_fixed_iter_3000.caffemodel
# ../models/center/center_step2_acc_iter_6000.caffemodel
# ../models/center/center_fixed_iter_3000.caffemodel

# ../models/smaller/alexnet_train_smaller_iter_500.caffemodel
# ../pretrained/alexnet/Submission_3.caffemodel

# ../models/smaller/alexnet_train_smaller_fer_iter_1540.caffemodel

# ../models/smaller/alexnet_train_smaller_fer_iter_1000.caffemodel
