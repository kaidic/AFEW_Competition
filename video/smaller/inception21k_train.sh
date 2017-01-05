#!/usr/bin/env sh
set -e

./build/tools/caffe train -solver ../scripts/smaller/inception21k_train_smaller_solver.prototxt -weights ../models/smaller/inception21k_smaller_fer_iter_300.caffemodel  -gpu 2 $@

# ../models/inception21k/inception21k_step2_iter_30000.caffemodel
# ../models/inception21k/inception21k_iter_1000.caffemodel 
# ../models/inception21k/inception21k_step2_addacc_iter_10000.caffemodel
# ../models/inception21k/inception21k_step3_addacc_iter_1500.caffemodel