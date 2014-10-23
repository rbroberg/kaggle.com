function entropy(x)
    y=x[1:size(x)[2]]
    freq=hist(y,[floor(minimum(y)):floor(maximum(y)+1)])
    probs=.000001+freq[2]/sum(freq[2])
    -1.*sum(probs .* log(2,probs))
end
