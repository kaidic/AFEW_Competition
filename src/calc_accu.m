function accuracy = calc_accu(probs, labels)
    [num, loc] = max(probs, [], 2);
    accuracy = sum(loc == labels) / length(labels);
end