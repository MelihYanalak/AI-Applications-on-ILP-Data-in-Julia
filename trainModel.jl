using Knet, IterTools, MLDatasets, CSV
using DataFrames
using LinearAlgebra

pathForTrainData= "allData.csv";
pathForTestData= "testData.csv";

struct Conv; w; b; end
Conv(w1,w2,nx,ny) = Conv(param(w1,w2,nx,ny), param0(1,1,ny,1))
(c::Conv)(x) = relu.(pool(conv4(c.w, x) .+ c.b))

struct Dense; w; b; f; end
Dense(i,o; f=identity) = Dense(param(o,i), param0(o), f)
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)

struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)




sample_df = CSV.read(pathForTrainData, DataFrame)
sample_df_test = CSV.read(pathForTestData, DataFrame)

matrix_form =   Matrix(sample_df)
matrix_form_test = Matrix(sample_df_test)


xTrn = Array{Float32, 3}(undef, 28, 28, 60000)
yTrn = Vector{Int64}(undef, 60000) 
xTst = Array{Float32, 3}(undef, 28, 28, 10000)
yTst = Vector{Int64}(undef, 10000) 

for i in 1:60000
  for j in 1:28
    for k in 1:28
      xTrn[j,k,i] = 0
    end
  end
end

col = 1
for i in 1:60000
  for j in 1:4#1 2 3 4 #1 2 3 4 5 6 7 8 9
    for k in 1:9
      xTrn[j,k,i] = matrix_form[i,(j-1)*9+k]
    end
  end
end

for i in 1:60000
  yTrn[i] = matrix_form[i,37]
end

for i in 1:10000
  for j in 1:28
    for k in 1:28
      xTst[j,k,i] = 0
    end
  end
end

col = 1
for i in 1:10000
  for j in 1:4 #1 2 3 4 #1 2 3 4 5 6 7 8 9
    for k in 1:9
      xTst[j,k,i] = matrix_form_test[i,(j-1)*9+k]
    end
  end
end

for i in 1:10000
  yTst[i] = matrix_form_test[i,37]
end

yTrn[yTrn.==0] .= 10
yTst[yTst.==0] .= 10
dTrn = minibatch(xTrn, yTrn, 100; xsize = (28,28,1,:))
dTst = minibatch(xTst, yTst, 100; xsize = (28,28,1,:))

model = Chain((Conv(5,5,1,20), Conv(5,5,20,50), Dense(800,500,f=relu), Dense(500,10)))

progress!(adam(model, ncycle(dTrn,3)))

accuracy(model,data=dTst)


