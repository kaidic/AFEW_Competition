%% Code for t selection
% Coded by Kaidi Cao

accu = zeros(1, 100);
tmp_count = 1;
for t = 0.01 : 0.01 : 1
    final_probs = video_probs + t * audio_probs;
    accu(tmp_count) = calc_accu(final_probs, labels);
    tmp_count = tmp_count + 1;
    endT
