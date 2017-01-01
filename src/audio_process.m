function ret_val = audio_process(filestr)
    tic;
    aviname = [filestr, '.avi'];
    seriname = strsplit(filestr, '/');
    seriname = seriname{end};
    wavname = ['tmp/', seriname, '.wav'];
    txtname = ['tmp/', seriname, '.txt'];
    system(['ffmpeg -i ',aviname,' -vn -b:a 128k -f wav ', wavname]);
    system(['openSMILE-2.2rc1/inst/bin/SMILExtract -C emobase2010.conf -I ', wavname,' -O ',txtname]);
    toc;
end