import numpy as np
import matplotlib.pyplot as plt
import DataLayerCUHK_noncaffe
import scipy.io as scio
import os, sys
import matplotlib.image as mpimg
import scipy

# Make sure that caffe is on the python path:
caffe_root = ''  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

model_choice = 1

if model_choice == 0:
    #### ALEXNET #### 227*227
    # MODEL_FILE = caffe_root + 'ReID/baseline_scnn/jstl_dgd_deploy.prototxt'
    MODEL_FILE = caffe_root + '../scripts/alexnet_lstm_deploy.prototxt'
    PRETRAINED = caffe_root + '../models/lstm/alexnet_lstm_final_iter_500.caffemodel'
    # PRETRAINED = caffe_root + '../models/center/alexnet_train_iter_300.caffemodel'
    imh = 227
    imw = 227

if model_choice == 1:
    MODEL_FILE = caffe_root + '../scripts/inception21k_lstm_deploy.prototxt'
    PRETRAINED = caffe_root + '../models/inception21k/inception21k_lstm_all_iter_6500.caffemodel'

    imh = 224
    imw = 224



print 'Using model: ', PRETRAINED
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(2)

test_dir = '../AFEW_Detect/Test/'

attributes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'] # for others

clip_len = 15


num_all = 0.0
num_right = 0.0

num_single = 0.0
num_right_single = 0.0

for i in xrange(len(attributes)):
    sub_dir = test_dir + attributes[i]
    print sub_dir
    vid_list = os.listdir(sub_dir)
    for vid_id in xrange(len(vid_list)):
        vid_folder = vid_list[vid_id]
        im_folder = test_dir + attributes[i] + '/' + vid_folder
        im_list = os.listdir(im_folder)
        n_im = len(im_list)

        # for a single video clip
        ims = np.zeros([clip_len,3,imh,imw])

        for i_im in xrange(clip_len):
            pos = int(round((n_im-1)*i_im / clip_len))

            im = mpimg.imread(test_dir + attributes[i] + '/' + vid_folder + '/' + im_list[pos]) 
            im = scipy.misc.imresize(im,[imh, imw])
            
            for ch in xrange(3):
                ims[i_im,ch,...] = im

        ims = ims - 129

        net.blobs['data'].reshape(clip_len, 3, imh, imw)
        net.blobs['clip_marker'].reshape(clip_len,1)
        clip_marker = np.ones([clip_len,1])
        clip_marker[0] = 0

        # print 'data shape', ims.shape
        # print 'clip_marker shape', clip_marker.shape


        feat = net.forward(data=ims, clip_marker=clip_marker)
        prob = feat['prob']
        prob = np.squeeze(prob)
        # print prob.shape

        num_single += clip_len
        for i_sin in xrange(clip_len):
            choice_single = np.where(prob[i_sin,:] == max(prob[i_sin,:]))
            choice_single = choice_single[0]
            choice_single = choice_single[0]
            # print choice_single
            if choice_single == i:
                num_right_single += 1



        avr_prob = np.mean(prob,0)
        # print avr_prob
        # print avr_prob
        choice = np.where(avr_prob == max(avr_prob))
        choice = choice[0]
        choice = choice[0]
        # print choice

        if choice == i:
            num_right += 1

        num_all += 1

print 'acc: ', num_right / num_all
print 'acc_single: ', num_right_single / num_single








'''

for iii in xrange(10):

    print 'Processing trial', iii
    
    ims = np.zeros([200,3,imh,imw])
    idx = 0
    for cam in xrange(2):
        for i in xrange(100):
            p_id = data_cuhk.true_testset_person_id[i] 
            im = data_cuhk.get_im_from_person_cam(p_id, cam)
            ims[idx,...] = im
            idx = idx + 1

    mini_size = 100

    start = 0
    net.blobs['data'].reshape(mini_size, 3, imh, imw)
    while start + mini_size < ims.shape[0]:
        print 'calc ', round((start*100. / ims.shape[0]),2),'%'
        mini_set = ims[start:start+mini_size, ...]
        
        feat = net.forward(data=mini_set)
        feature[iii, start:start+mini_size, ...] = feat['fc7_bn']
        start += mini_size

    mini_set = ims[start:ims.shape[0],...]
    net.blobs['data'].reshape(mini_set.shape[0], 3, imh, imw)
    feat = net.forward(data=mini_set)
    feature[iii, start:ims.shape[0], ...] = feat['fc7_bn']


# out_file = 'ReID/result/exp3/feat_cos_con_' + str(it+1) + '00.mat'
out_file = 'ReID/result/exp13/feat_dgd.mat'
scio.savemat(out_file, {'feature': feature}) 

print 'feature shape:', feature.shape
print 'save to', out_file
'''