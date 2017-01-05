#!/usr/bin/env sh
set -e

./build/tools/caffe train -solver ../scripts/inception21k_train_solver_fc2.prototxt -weights ../pretrained/inception21k/Inception21k.caffemodel  -gpu 0 $@