%% Predicting emotions using video info
% Coded by Kaidi Cao

function ret_val = video_process(filestr)

    %% Initialization
    %path to be added
    caffe_path = './caffe/matlab';
    mtcnn_path = './MTCNN_face_detection_alignment/code/codes/MTCNNv2';
    pdollar_toolbox_path = './toolbox';
    caffe_model_path = './MTCNN_face_detection_alignment/code/codes/MTCNNv2/model';
    addpath(genpath(caffe_path));
    addpath(genpath(mtcnn_path));
    addpath(genpath(pdollar_toolbox_path));
    caffe.reset_all();
    %use cpu
    %caffe.set_mode_cpu();
    gpu_id=0;
    caffe.set_mode_gpu();	
    caffe.set_device(gpu_id);

    %three steps's threshold
    threshold = [0.6 0.7 0.7];

    %scale factor
    factor=0.709;

    %load caffe models
    prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
    model_dir = strcat(caffe_model_path,'/det1.caffemodel');
    PNet=caffe.Net(prototxt_dir,model_dir,'test');
    prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
    model_dir = strcat(caffe_model_path,'/det2.caffemodel');
    RNet=caffe.Net(prototxt_dir,model_dir,'test');	
    prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
    model_dir = strcat(caffe_model_path,'/det3.caffemodel');
    ONet=caffe.Net(prototxt_dir,model_dir,'test');
    prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
    model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
    LNet=caffe.Net(prototxt_dir,model_dir,'test');

    % load mat file
    
    [filePath, fileName, fileExt]   =   fileparts(filestr);

    % load decoded audio and video frames from .mat file.
    load(fullfile(filePath, [fileName, '.mat']));
    
    %% Face Detection
    img_size = 224;
    detected_imgs = zeros(img_size, img_size, 3, ceil(length(video.frames)/2));
    detected = 0;
    for i = 1: 2 :length(video.frames)
        img = video.frames(i).cdata;
        minl = min([size(img,1) size(img,2)]);
        minsize = fix(minl*0.1);

        [boundingboxes, points] = detect_face(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);

        % find the bounding box with max area
        numbox=size(boundingboxes,1);
        max_area = 0;
        max_index = 0;
        for j = 1 : numbox
             area = (boundingboxes(j, 3) - boundingboxes(j, 1)) * (boundingboxes(j, 4) - boundingboxes(j, 2));
             if area > max_area
                 max_area = area;
                 max_index = j; 
             end
        end

        % crop image
        if max_index > 0
            shape = size(img);
            bbox = rerec(boundingboxes(max_index, :)); 
            bbox(1) = max(1, ceil(bbox(1)));
            bbox(2) = max(1, ceil(bbox(2)));
            bbox(3) = min(shape(2), floor(bbox(3)));
            bbox(4) = min(shape(1), floor(bbox(4)));
            crop_face = imcrop(img, [bbox(1:2) bbox(3:4)-bbox(1:2)]);
            crop_face = imresize(crop_face, [img_size, img_size]);
            crop_face = rgb2gray(crop_face);
            crop_face = crop_face';
            detected = detected + 1;
            detected_imgs(:, :, : ,detected) = repmat(reshape(crop_face, [img_size, img_size, 1]), [1, 1, 3]);
        end           
    end
    
    % No valid detection was made
    if detected == 0
        ret_val = -1;
        return;
    end
    
    % get clips we'll use
    clip_len = 15;
    clip_imgs = zeros(img_size, img_size, 3, clip_len);
    for i = 1: clip_len
        pos = round((detected - 1) * (i - 1) / clip_len) + 1;
        clip_imgs(:, :, :, i) = detected_imgs(: , :, :, pos);
    end

    clip_imgs = clip_imgs - 129;
    clip_marker = ones([1, 1, 1, clip_len]);
    clip_marker(1) = 0;

    %% alex_lstm
    prototxt_dir = './model/inception21k_lstm_deploy.prototxt';
    model_dir = './model/inception21k_lstm_all_iter_3600.caffemodel';
    net = caffe.Net(prototxt_dir, model_dir, 'test');

    input_blob = {clip_imgs, clip_marker};
    prob = net.forward(input_blob);
    prob = prob{1};
    
    % mean pooling
    ret_val = mean(prob, 3)';
end
