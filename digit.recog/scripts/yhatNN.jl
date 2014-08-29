# http://blog.yhathq.com/posts/julia-neural-networks.html
# Pkg.clone("https://github.com/EricChiang/ANN.jl.git")

using DataFrames
train_df = readtable("/projects/kaggle.com/digit.recog/data/train.csv",separator=',');
size(train_df)
names(train_df)

# Get value counts of "label" classes
_, count = hist(train_df[:label])
class = sort(unique(train_df[:label]))
value_counts = DataFrame(count=count, class=class)

#=
'''
using Gadfly
# Draw a plot to your browser
p = plot(value_counts,
         x="class", y="count",
         Geom.bar(), Guide.title("Class distributions (\"quality\")")
)
draw(PNG(14cm,10cm),p)
'''
=#
function confusion_matrix(y_true::Array{Int64,1},y_pred::Array{Int64,1})
    # Generate confusion matrix
    classes = sort(unique([unique(y_true),unique(y_pred)]))
    cm = zeros(Int64,length(classes),length(classes))
	
    for i in 1:length(y_test)
        # translate label to index
        true_class = findfirst(classes,y_test[i])
        pred_class = findfirst(classes,y_pred[i])
        # pred class is the row, true class is the column
        cm[pred_class,true_class] += 1
    end
    cm
end


# "it's like SciPy had a love child with R"
y = array(train_df[:label]);
X = array(train_df[:,2:end]);

# Generate train/test split 75/25
n = length(y);
is_train = shuffle([1:n] .> floor(n * .90));

X_train,X_test = X[is_train,:],X[!is_train,:];
y_train,y_test = y[is_train],y[!is_train];

# print stuff to the user
println("Total number of observations: ",n)
println("Training set size: ", sum(is_train))
println("Test set size: ", sum(!is_train))

using ANN

# The larger the hidden layer, the more likely the model is to overfit. 
# The smaller the hidden layer, the more likely the model is to overgeneralize.
nhl=int(sqrt(size(X_test,2)))
ann = ArtificialNeuralNetwork(size(X_test,2))

# nope!
# fit!(ann,X_train,y_train,epochs=30,alpha=0.1,lambda=1e-5)
# typeof(X_train)  # Array{Int64,2}

X_train=convert(Array{Float64,2},X_train);

fit!(ann,X_train,y_train,epochs=30,alpha=0.1,lambda=1e-5);

X_test=convert(Array{Float64,2},X_test);
y_proba = predict(ann,X_test);

y_pred = Array(Int64,length(y_test));

for i in 1:length(y_test)
    # must translate class index to label
    y_pred[i] = ann.classes[indmax(y_proba[i,:])]
end

println("Prediction accuracy: ",mean(y_pred .== y_test))

confusion_matrix(y_test,y_pred)

# ============================================================
# this is all for scaling, not necessary on this data set
# ============================================================
#=
'''
# Kinda like a C struct
type StandardScalar
    mean::Vector{Float64}
    std::Vector{Float64}
end

# Helper function to initialize an empty scalar
function StandardScalar()
    StandardScalar(Array(Float64,0),Array(Float64,0))
end

# Compute mean and standard deviation of each column
function fit_std_scalar!(std_scalar::StandardScalar,X::Array{I64})
    n_rows, n_cols = size(X_test)
    std_scalar.std = zeros(n_cols)
    std_scalar.mean = zeros(n_cols)
    # for loops are fast again!
    for i = 1:n_cols
        std_scalar.mean[i] = mean(X[:,i])
        std_scalar.std[i] = std(X[:,i])
    end
end

function transform(std_scalar::StandardScalar,X::Array{Int64,2})
    (X .- std_scalar.mean') ./ std_scalar.std' # broadcasting fu
end

# fit and transform in one function
function fit_transform!(std_scalar::StandardScalar,X::Array{Int64,2})
    fit_std_scalar!(std_scalar,X)
    transform(std_scalar,X)
end

std_scalar = StandardScalar()

n_rows, n_cols = size(X_test)

# what do columns look like before scaling?
println("Column means before scaling:")
for i = 1:n_cols
    @printf("%0.3f ",(mean(X_test[:,i])))
end

X_train = fit_transform!(std_scalar,X_train)
X_test = transform(std_scalar,X_test)

# ... and after scaling?
println("\nColumn means after scaling:")
for i = 1:n_cols
    @printf("%0.3f ",(mean(X_test[:,i])))
end

# there is a lot of NaNs
'''
=#
