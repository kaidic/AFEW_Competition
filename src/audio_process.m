%% Predicting emotions using audio info
% Coded by Jingkai Yan, modified by Kaidi Cao

function ret_val = audio_process(filestr)

    svmpath = 'libsvm/matlab';
    addpath(genpath(svmpath));
    %% Extract features
    aviname = filestr;
    seriname = strsplit(filestr, '/');
    seriname = seriname{end};
    seriname = seriname(1:end-4);
    wavname = ['tmp/', seriname, '.wav'];
    txtname = ['tmp/', seriname, '.txt'];
    system(['ffmpeg -i ',aviname,' -vn -b:a 128k -f wav ', wavname]);
    system(['openSMILE-2.2rc1/inst/bin/SMILExtract -C emobase2010.conf -I ', wavname,' -O ',txtname]);
    
    %% Load features
    fin = fopen(txtname, 'r');
    for j = 1:1589
        fgetl(fin);
    end
    line = fgetl(fin);
    line = line(10:end);
    feat = textscan(line,'%f','Delimiter',',');
    feat = feat{1}';
    fclose(fin);
    
    %% Predict
    load modelAudio.mat;
    [predict_label, accuracy, dec_values] = svmpredict(1, feat, model);
    prob = zeros(1, 7);
    id = [1 2 3 4 5 6;
      1 7 8 9 10 11;
      2 7 12 13 14 15;
      3 8 12 16 17 18;
      4 9 13 16 19 20;
      5 10 14 17 19 21;
      6 11 15 18 20 21];
    weight = [1 1 1 1 1 1;
              -1 1 1 1 1 1;
              -1 -1 1 1 1 1;
              -1 -1 -1 1 1 1;
              -1 -1 -1 -1 1 1;
              -1 -1 -1 -1 -1 1;
              -1 -1 -1 -1 -1 -1];
    for i = 1:7
        prob(i) = dec_values(:,id(i,:)) * weight(i,:)';
    end
    prob = exp(prob);
    
    prob = prob ./ sum(prob);
    ret_val = prob;
end