function res = audio_preprocess(filestr)
    aviname = [filestr, '.avi'];
    wavname = [filestr, '.wav'];
    txtname = [filestr, '.txt'];
    system(['ffmpeg -i ',aviname,' -vn -b:a 128k -f wav ', wavname]);
    system(['openSMILE-2.1.0/inst/bin/SMILExtract -C emobase2010.conf -I ', wavname,' -O ',txtname]);
end