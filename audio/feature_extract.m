featureAudio = [];
labelAudio = [];
folders = {'Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'};

for fid = 1:7

    dirs = dir([folders{fid},'\*.txt']);
    dirs = struct2cell(dirs);
    names = dirs(1,:);
    N = length(names);

    featureLocal = zeros(N,1583);
    labelLocal = ones(N,1) * (fid-1);

    for i = 1:N
        name = names{i};
        fp = fopen(name,'r');
        for j = 1:1589
            fgetl(fp);
        end
        line = fgetl(fp);
        line = line(10:end);
        c = textscan(line,'%f','Delimiter',',');
        featureLocal(i,:) = c{1}';
        fclose(fp);
    end
    featureAudio = [featureAudio;featureLocal];
    labelAudio = [labelAudio;labelLocal];    
end

save dataAudio.mat featureAudio labelAudio