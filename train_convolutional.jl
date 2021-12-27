using Knet, IterTools, MLDatasets, CSV
using DataFrames
using LinearAlgebra


pathForTrainData= "allData_10var.csv";
pathForTestData= "testData_10var.csv";

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
  for j in 1:4
    for k in 1:14
      xTrn[j,k,i] = matrix_form[i,(j-1)*14+k]
    end
  end
end

numberOfMethod_1 = 0
numberOfMethod_2 = 0
numberOfMethod_3 = 0
for i in 1:60000
  yTrn[i] = matrix_form[i,57]
  if yTrn[i] == 1.0
    global numberOfMethod_1
    numberOfMethod_1 = numberOfMethod_1 + 1
  elseif yTrn[i] == 2.0
    global numberOfMethod_2
    numberOfMethod_2 = numberOfMethod_2 + 1
  elseif yTrn[i] == 3.0
    global numberOfMethod_3
    numberOfMethod_3 = numberOfMethod_3 + 1
  end
end
println(numberOfMethod_1)
println(numberOfMethod_2)
println(numberOfMethod_3)
for i in 1:10000
  for j in 1:28
    for k in 1:28
      xTst[j,k,i] = 0
    end
  end
end

col = 1
for i in 1:10000
  for j in 1:4 
    for k in 1:14
      xTst[j,k,i] = matrix_form_test[i,(j-1)*14+k]
    end
  end
end

for i in 1:10000
  yTst[i] = matrix_form_test[i,57]
end

yTrn[yTrn.==0] .= 10
yTst[yTst.==0] .= 10



function train(model, data, optim)
    for (x,y) in data
        grads = lossgradient(model,x,y)
        update!(model, grads, optim)
    end
end
function predict(w,x0)
    x1 = pool(relu.(conv4(w[1],x0) .+ w[2]))
    x2 = pool(relu.(conv4(w[3],x1) .+ w[4]))
    x3 = relu.(w[5]*mat(x2) .+ w[6])
    return w[7]*x3 .+ w[8]
end

loss(w,x,ygold) = nll(predict(w,x), ygold)

lossgradient = grad(loss)

w = Any[ xavier(Float32,5,5,1,20),  zeros(Float32,1,1,20,1),
         xavier(Float32,5,5,20,50), zeros(Float32,1,1,50,1),
         xavier(Float32,500,800),   zeros(Float32,500,1),
         xavier(Float32,10,500),    zeros(Float32,10,1) ]

dTrn = minibatch(xTrn, yTrn, 100; xsize = (28,28,1,:))
dTst = minibatch(xTst, yTst, 100; xsize = (28,28,1,:))

o = optimizers(w, Adam)
println((:epoch, 0, :trn, accuracy(w,dTrn,predict), :tst, accuracy(w,dTst,predict)))
for epoch=1:10
    train(w, dTrn, o)
    println((:epoch, epoch, :trn, accuracy(w,dTrn,predict), :tst, accuracy(w,dTst,predict)))
end