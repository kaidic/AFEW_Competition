function    ret =   fun_process(filestr)

    [filePath, fileName, fileExt]   =   fileparts(filestr);
    disp(filestr);

    % load decoded audio and video frames from .mat file.
    load(fullfile(filePath, [fileName, '.mat']));

    
    
    ret_set =   {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
    if (exist('fun_classification.m', 'file'))
        ret_val =   fun_classification(filestr);
    else
        warning('No available classification function. Random result returned.');
        ret_val =   randi([0, 6]);
    end
    ret =   ret_set{ret_val + 1};

end

