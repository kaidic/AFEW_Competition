%% Predicting emotions using both video info and audio info
% Coded by Kaidi Cao

function final_ret = fun_classification(filestr)
    
    %% audio part
    audio_prob = audio_process(filestr);
    
    %% video part
    video_prob = video_process(filestr);
    
    %% prob combination
    % no face detected
    if length(video_prob) == 1
        final_prob = audio_prob;
    else
        final_prob = video_prob + 0.22 * audio_prob;
    end
    
    [num, loc] = max(final_prob);
    final_ret = loc - 1;
end
