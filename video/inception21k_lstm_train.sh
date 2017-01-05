#!/usr/bin/env sh
set -e

./build/tools/caffe train -solver ../scripts/inception21k_lstm_solver.prototxt -weights ../models/inception21k/inception21k_alldata_iter_1000.caffemodel -gpu 0 $@

# ../models/inception21k/inception21k_step2_iter_30000.caffemodel
# ../models/inception21k/inception21k_iter_1000.caffemodel 
# ../models/inception21k/inception21k_step2_addacc_iter_10000.caffemodel
# ../models/inception21k/inception21k_step3_addacc_iter_1500.caffemodel