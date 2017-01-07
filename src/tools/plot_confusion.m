%% Code that can plot confusion matrix
% Coded by Kaidi Cao

clabels = zeros(7, 379);
for i = 1 : 379
    clabels(labels(i), i) = 1 ;
end
cloc = zeros(7, 379);
for i = 1 : 379
    cloc(loc(i), i) = 1 ;
end

plotconfusion(clabels, cloc)
