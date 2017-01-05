# Code by Lin Ji.
'''
This is a self-defined layer for preparing data into the network
This is for dataset WARD
'''

import caffe
import numpy as np
import scipy
import yaml
# import cv2 # opencv
import scipy.io as scio
import time
import math, os
import matplotlib.image as mpimg


class DataLayer(caffe.Layer):
    '''
    def get_next_train_batch(self):
        ims = np.zeros([self.clip_len * self.batch_size, 3, self.imh, self.imw])
        label = np.zeros([self.clip_len, self.batch_size])
        inds = np.random.permutation(self.num_train)
        inds = inds[0:self.batch_size]
        for i in xrange(self.batch_size):
            ind = inds[i]
            im_dir = self.train_dir[ind]
            label[:,i] = self.train_label[ind]

            im_list = os.listdir(self.train_root + im_dir)
            n_im = len(im_list)
            for i_im in xrange(self.clip_len):
                pos = int(round((n_im-1)*i_im / self.clip_len))

                im = mpimg.imread(self.train_root + im_dir + '/' + im_list[pos]) 
                im = scipy.misc.imresize(im,[self.imh, self.imw])

                ims[i_im * self.batch_size + i,0,...] = im
                ims[i_im * self.batch_size + i,1,...] = im
                ims[i_im * self.batch_size + i,2,...] = im

        ims -= self.mean

        clip_marker = np.ones([self.clip_len, self.batch_size])
        clip_marker[0,:] = 0

        self.net.blobs['data'].reshape(self.clip_len, 3, 224, 224)
        feat = np.zeros([self.batch_size*self.clip_len,1024])

        for i in xrange(self.batch_size):
            ims_part = ims[i*self.clip_len:(i+1)*self.clip_len,...]
            feat2 = self.net.forward(data=ims_part)
            feat_inner = feat2['global_pool']
        
            # print feat.shape
            feat[i*self.clip_len:(i+1)*self.clip_len,:]=np.squeeze(feat_inner)

        blobs = {'data': feat,
                'label': label,
                'clip_marker': clip_marker
                }
        return blobs


    def get_next_val_batch(self):
        ims = np.zeros([self.clip_len * self.batch_size, 3, self.imh, self.imw])
        label = np.zeros([self.clip_len, self.batch_size])
        inds = np.random.permutation(self.num_val)
        inds = inds[0:self.batch_size]
        for i in xrange(self.batch_size):
            ind = inds[i]
            im_dir = self.val_dir[ind]
            label[:,i] = self.val_label[ind]

            im_list = os.listdir(self.val_root + im_dir)
            n_im = len(im_list)
            for i_im in xrange(self.clip_len):
                pos = int(round((n_im-1)*i_im / self.clip_len))

                im = mpimg.imread(self.val_root + im_dir + '/' + im_list[pos]) 
                im = scipy.misc.imresize(im,[self.imh, self.imw])

                ims[i_im * self.batch_size + i,0,...] = im
                ims[i_im * self.batch_size + i,1,...] = im
                ims[i_im * self.batch_size + i,2,...] = im

        ims -= self.mean

        clip_marker = np.ones([self.clip_len, self.batch_size])
        clip_marker[0,:] = 0
        
        self.net.blobs['data'].reshape(self.clip_len, 3, 224, 224)
        feat = np.zeros([self.batch_size*self.clip_len,1024])

        for i in xrange(self.batch_size):
            ims_part = ims[i*self.clip_len:(i+1)*self.clip_len,...]
            feat2 = self.net.forward(data=ims_part)
            feat_inner = feat2['global_pool']
        
            # print feat.shape
            feat[i*self.clip_len:(i+1)*self.clip_len,:]=np.squeeze(feat_inner)

        blobs = {'data': feat,
                'label': label,
                'clip_marker': clip_marker
                }
        return blobs

    '''

    def get_next_train_batch(self):
        ims = np.zeros([self.clip_len * self.batch_size, 3, self.imh, self.imw])
        label = np.zeros([self.clip_len, self.batch_size])
        inds = np.random.permutation(self.num_train)
        inds = inds[0:self.batch_size]

        feat = self.train_feat[inds,...]

        for i in xrange(self.batch_size):
            ind = inds[i]
            label[:,i] = self.train_label[ind]


        clip_marker = np.ones([self.clip_len, self.batch_size])
        clip_marker[0,:] = 0

        # print feat


        blobs = {'data': feat,
                'label': label,
                'clip_marker': clip_marker
                }
        return blobs



    def get_next_val_batch(self):
        ims = np.zeros([self.clip_len * self.batch_size, 3, self.imh, self.imw])
        label = np.zeros([self.clip_len, self.batch_size])
        inds = np.random.permutation(self.num_val)
        inds = inds[0:self.batch_size]

        feat = self.val_feat[inds,...]

        for i in xrange(self.batch_size):
            ind = inds[i]
            label[:,i] = self.val_label[ind]


        clip_marker = np.ones([self.clip_len, self.batch_size])
        clip_marker[0,:] = 0


        blobs = {'data': feat,
                'label': label,
                'clip_marker': clip_marker
                }
        return blobs



    def setup(self, bottom, top):
        layers_params = yaml.load(self.param_str)
        self.stage = layers_params['stage'] # train or test
        self.imh = layers_params['imh']
        self.imw = layers_params['imw']


        self.clip_len = 15
        self.batch_size = layers_params['batch_size']

        self.mean = 129
        
        self.caffe_root = ''
        self.train_root = '../AFEW_Detect/'
        self.val_root = '../AFEW_Detect/Val/'

        ''' 
        self.train_dir = []
        self.train_label = []
        self.val_dir = []
        self.val_label = []
        '''

        mat = scio.loadmat('../lstm_feat.mat')

        self.train_feat = mat['train_feat']
        self.val_feat = mat['val_feat']
        self.train_label = np.squeeze(mat['train_label'])
        self.val_label = np.squeeze(mat['val_label'])


        
        '''
        f_train = open(self.train_root + "clip_train_test.txt", "r")  
        while True:  
            line = f_train.readline()  
            if line:  
                line=line.strip()
                self.train_label.append(int(line[-1:]))
                self.train_dir.append(line[0:-2])
                print line[0:-2]
            else:
                break
        f_train.close()

        f_val = open(self.val_root + "clip_val.txt", "r")  
        while True:  
            line = f_val.readline()  
            if line:  
                line=line.strip()
                self.val_label.append(int(line[-1:]))
                self.val_dir.append(line[0:-2])
            else:  
                break
        f_val.close()

        print len(self.train_dir), len(self.val_dir)

        self.num_train = len(self.train_dir)
        self.num_val = len(self.val_dir)

        '''

        self.num_train = self.train_feat.shape[0]
        self.num_val = self.val_feat.shape[0]


        self.name_to_top_map = {
            'data': 0,
            'label': 1,
            'clip_marker': 2
            }
        
        
        # data blobs: holds a batch of N_images, each with 3 channels
        # The height and width (48x48) are values for ViPeR
        top[0].reshape(self.clip_len * self.batch_size, 1024)
        # labels blob: it's a binary 
        top[1].reshape(self.clip_len, self.batch_size)

        top[2].reshape(self.clip_len * self.batch_size)

        MODEL_FILE = '../scripts/inception21k_feat.prototxt'
        # PRETRAINED = caffe_root + '../models/inception21k/inception21k_step4_decay_iter_300.caffemodel'
        PRETRAINED = '../models/inception21k/inception21k_alldata_iter_1000.caffemodel'
        self.net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
        caffe.set_mode_gpu()
        caffe.set_device(2)

    def forward(self, bottom, top):
    # get blobs and copy them into this layer's top blob vector
        if self.stage == 'train':
            blobs = self.get_next_train_batch()
        else:
            blobs = self.get_next_val_batch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self.name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blob
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)


            # print 'blob shape:', blob.shape

    def backward(self, top, propagate_down, bottom):
         # This layer does not perform backward method
        pass

    def reshape(self, bottom, top):
    # reshaping is done in the call to forward
        pass


