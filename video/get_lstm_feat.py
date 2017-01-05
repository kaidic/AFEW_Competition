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

model_choice = 2

if model_choice == 0:
    #### ALEXNET #### 227*227
    # MODEL_FILE = caffe_root + 'ReID/baseline_scnn/jstl_dgd_deploy.prototxt'
    MODEL_FILE = caffe_root + '../scripts/alexnet_deploy.prototxt'
    # PRETRAINED = caffe_root + '../pretrained/alexnet/Submission_3.caffemodel'
    PRETRAINED = caffe_root + '../models/alexnet/alexnet_train_alldata_iter_1300.caffemodel'
    imh = 227
    imw = 227

    attributes = ['Fear','Sad','Happy','Surprise','Disgust','Angry','Neutral'] # for alexnet 

elif model_choice == 1:
    #### CENTERLOSS #### 112*96
    MODEL_FILE = caffe_root + '../scripts/face_deploy.prototxt'
    # PRETRAINED = caffe_root + '../models/center/center_step3_addacc_iter_1500.caffemodel'
    # PRETRAINED = caffe_root + '../pretrained/centerloss/face_model.caffemodel'
    PRETRAINED = caffe_root + '../models/center/center_step3_addacc_iter_1000.caffemodel'
    imh = 112
    imw = 96

    attributes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'] # for others

elif model_choice == 2:
    #### INCEPTION #### 224*224
    MODEL_FILE = caffe_root + '../scripts/inception21k_feat.prototxt'
    # PRETRAINED = caffe_root + '../models/inception21k/inception21k_step4_decay_iter_300.caffemodel'
    PRETRAINED = caffe_root + '../models/inception21k/inception21k_alldata_iter_1000.caffemodel'

    imh = 224
    imw = 224

    attributes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'] # for others

#### FOR SMALLER NETS
elif model_choice == 3:
    MODEL_FILE = '../scripts/smaller/inception21k_deploy_smaller.prototxt'
    PRETRAINED = caffe_root + '../models/smaller/inception21k_smaller_afew_iter_5000.caffemodel'

    imh = 128
    imw = 128

    attributes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'] # for others

elif model_choice == 4:
    MODEL_FILE = '../scripts/smaller/alexnet_deploy_smaller.prototxt'
    PRETRAINED = caffe_root + '../models/smaller/alexnet_train_smaller_alldata_iter_2500.caffemodel'

    imh = 117
    imw = 117

    attributes = ['Fear','Sad','Happy','Surprise','Disgust','Angry','Neutral'] # for others




net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(2)


print 'Using model: ', PRETRAINED
print 'imh: ', imh, 'imw', imw

caffe_root = ''
train_root = '../AFEW_Detect/'
val_root = '../AFEW_Detect/Val/'

train_dir = []
train_label = []
val_dir = []
val_label = []

clip_len = 15
mean = 129

f_train = open(train_root + "clip_train_test.txt", "r")  
while True:  
    line = f_train.readline()  
    if line:  
        line=line.strip()
        train_label.append(int(line[-1:]))
        train_dir.append(line[0:-2])
        print line[0:-2]
    else:
        break
f_train.close()

f_val = open(val_root + "clip_val.txt", "r")  
while True:  
    line = f_val.readline()  
    if line:  
        line=line.strip()
        val_label.append(int(line[-1:]))
        val_dir.append(line[0:-2])
    else:  
        break
f_val.close()


train_feat = np.zeros([len(train_dir),15,1024])
val_feat = np.zeros([len(val_dir),15,1024])


for i in xrange(len(train_dir)):
    im_dir = train_dir[i]
    im_list = os.listdir(train_root + im_dir)

    n_im = len(im_list)
    ims = np.zeros([clip_len, 3, imh, imw])
    for i_im in xrange(clip_len):
        pos = int(round((n_im-1)*i_im / clip_len))

        im = mpimg.imread(train_root + im_dir + '/' + im_list[pos]) 
        im = scipy.misc.imresize(im,[imh, imw])

        ims[i_im,0,...] = im
        ims[i_im,1,...] = im
        ims[i_im,2,...] = im

    ims -= mean

    net.blobs['data'].reshape(clip_len, 3, 224, 224)
    feat = net.forward(data=ims)
    feat_inner = feat['global_pool']

    # print feat.shape
    train_feat[i,...] = np.squeeze(feat_inner)
    print i 


for i in xrange(len(val_dir)):
    im_dir = val_dir[i]
    im_list = os.listdir(val_root + im_dir)

    n_im = len(im_list)
    ims = np.zeros([clip_len, 3, imh, imw])
    for i_im in xrange(clip_len):
        pos = int(round((n_im-1)*i_im / clip_len))

        im = mpimg.imread(val_root + im_dir + '/' + im_list[pos]) 
        im = scipy.misc.imresize(im,[imh, imw])

        ims[i_im,0,...] = im
        ims[i_im,1,...] = im
        ims[i_im,2,...] = im

    ims -= mean

    net.blobs['data'].reshape(clip_len, 3, 224, 224)
    feat = net.forward(data=ims)
    feat_inner = feat['global_pool']

    # print feat.shape
    val_feat[i,...] = np.squeeze(feat_inner)
    print i 

out_file = '../lstm_feat.mat'
scio.savemat(out_file, {'train_feat': train_feat, 'val_feat': val_feat, 'train_label': train_label, 'val_label': val_label}) 


print 'feature shape:', train_feat.shape
print 'save to', out_file



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