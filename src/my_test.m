%% My test function on whole validation dataset
% Coded by Kaidi Cao

dir_prefix = '/data/AFEW/Val/';
emotion_class = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
audio_probs = [];
video_probs = [];
labels = [];
for tmp_class = 1 : 7
    films = dir([dir_prefix, emotion_class{tmp_class}, '/*.mat']);
    for f_id = 1 : length(films)
        full_dir = [dir_prefix, emotion_class{tmp_class}, '/', films(f_id).name];
        full_dir = full_dir(1: end-4);
        tic
        [audio_prob, video_prob] = fun_classification(full_dir);
        toc
        if length(video_prob) ~= 1
            video_probs = [video_probs; video_prob];
            audio_probs = [audio_probs; audio_prob];
            labels = [labels; tmp_class];
        end
    end
end