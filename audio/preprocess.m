folders = {'Angry\','Disgust\','Fear\','Happy\','Neutral\','Sad\','Surprise\'};
for fid = 1%:7
    cd(folders{fid})
    dirs = dir('*.avi');
    dirs = struct2cell(dirs);
    names = dirs(1,:);
    N = length(names);

    for i = 1:N
        aviname = names{i};
        wavname = [aviname(1:9),'.wav'];
        txtname = [aviname(1:9),'.txt'];
        system(['..\ffmpeg -i ',aviname,' -vn -b:a 128k -f wav ',wavname])
        system(['..\SMILExtract_Release.exe -C ..\emobase2010.conf -I ',wavname,' -O ',txtname])
    end
    system('cd ..')
end
