model = svmtrain(labelAudio,featureAudio, '-t 1 -d 1');
[predict_label, accuracy, dec_values] =svmpredict(labelAudioVal,featureAudioVal, model);

prob = zeros(length(labelAudioVal),7);
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
    prob(:,i) = dec_values(:,id(i,:)) * weight(i,:)';
end
prob = exp(prob);
for i = 1:length(labelAudioVal)
    prob(i,:) = prob(i,:) / sum(prob(i,:));
end

predict_label_prob = zeros(length(labelAudioVal),1);
for i = 1:length(labelAudioVal)
    for j = 2:7
        if prob(i,j)>prob(i,predict_label_prob(i)+1)
            predict_label_prob(i) = j-1;
        end
    end
end
