% Modified by Kaidi Cao

function    ret =   fun_process(filestr)

    ret_set =   {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
    if (exist('fun_classification.m', 'file'))
        ret_val =   fun_classification(filestr);
    else
        warning('No available classification function. Random result returned.');
        ret_val =   randi([0, 6]);
    end
    ret =   ret_set{ret_val + 1};

end

