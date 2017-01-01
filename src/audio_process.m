function ret_val = audio_process(filestr)
    tic
    svmpath = 'libsvm/matlab';
    addpath(genpath(svmpath));
    %% Extract features
    aviname = [filestr, '.avi'];
    seriname = strsplit(filestr, '/');
    seriname = seriname{end};
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
    ret_val = dec_values;
    toc
end