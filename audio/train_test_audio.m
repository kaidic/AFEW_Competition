model = svmtrain(labelAudio,featureAudio);
[predict_label, accuracy, dec_values] =svmpredict(labelAudioVal,featureAudioVal, model);
