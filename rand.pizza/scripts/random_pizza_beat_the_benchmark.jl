# caveat: I am just learning Julia and this is an exercise.

# rbroberg: 20140821

using DataFrames
using DecisionTree
using JSON

datadir="/projects/kaggle.com/rand.pizza/data/"
submitdir="/projects/kaggle.com/rand.pizza/submissions/"


# http://rosettacode.org/wiki/Flatten_a_list#Julia
function flat(A)
   result = {}
   grep(a) = for x in a 
               isa(x,Array) ? grep(x) : push!(result,x)
             end
   grep(A)
   result
end

jtrain=JSON.parsefile(datadir*"train.json");

# this doesn't help much,
# just a 1D data frame of dictionaries
# dftrain=DataFrame(jtrain);
# dftrain

# -----------------------------------------------------------------------------
# extract days since first post
# -----------------------------------------------------------------------------
nrow=size(jtrain)[1];
tarr=zeros((nrow,2));
[tarr[i,:]=[jtrain[i]["requester_days_since_first_post_on_raop_at_request"],
	jtrain[i]["requester_received_pizza"]*1] for i in 1:nrow];
dftrain=DataFrame(tarr);

mean(dftrain[dftrain[:,2].==0,1]) # 12.45
mean(dftrain[dftrain[:,2].==1,1]) # 28.56

median(dftrain[dftrain[:,2].==0,1]) # 0.0 
median(dftrain[dftrain[:,2].==1,1]) # 0.0

# not good enough by itself

# -----------------------------------------------------------------------------
# sentiment analysis
# -----------------------------------------------------------------------------

# extract words from title and text
# create dictionary
# find pizza prob for each word
# predict by summing probs


# extract words from title and text
nrow=size(jtrain)[1];
zas=[jtrain[i]["requester_received_pizza"]*1 for i in 1:nrow];
words=[split(jtrain[i]["request_title"]*" "*jtrain[i]["request_text"]) for i in 1:nrow];

# create dictionary
#TO-DO: clean up [Request], punctation, trailing ...
wordbag=flat(words);
worduniq=Set(wordbag);
length(worduniq)

