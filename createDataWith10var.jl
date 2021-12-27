using JuMP
using Gurobi
using BenchmarkTools
using CSV
using DataFrames


function generateProblem(numberOfData)
    i = 0
    opt = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
    model = Model(opt)
    #while i <= 0
    @variable(model,0 <= x[1:10] <= 100)
    coeffList = rand((0:100), 10)
    @objective(model, Max, coeffList[1]*x[1]+coeffList[2]*x[2]+coeffList[3]*x[3]+coeffList[4]*x[4]+coeffList[5]*x[5]+coeffList[6]*x[6]+coeffList[7]*x[7]+coeffList[8]*x[8]+coeffList[9]*x[9]+coeffList[10]*x[10])
    feasibleFound = 0

    coeffListForConstraints = rand((0:100), 10)
    rightHandSide = rand(0:1000)
    operatorChoice = rand(1:3)
    if operatorChoice == 1
        @constraint(model, con, coeffListForConstraints[1]*x[1]+coeffListForConstraints[2]*x[2]+coeffListForConstraints[3]*x[3]+coeffListForConstraints[4]*x[4]+coeffListForConstraints[5]*x[5]+coeffListForConstraints[6]*x[6]+coeffListForConstraints[7]*x[7]+coeffListForConstraints[8]*x[8]+coeffListForConstraints[9]*x[9]+coeffListForConstraints[10]*x[10] == rightHandSide )
    elseif operatorChoice == 2
        @constraint(model, con, coeffListForConstraints[1]*x[1]+coeffListForConstraints[2]*x[2]+coeffListForConstraints[3]*x[3]+coeffListForConstraints[4]*x[4]+coeffListForConstraints[5]*x[5]+coeffListForConstraints[6]*x[6]+coeffListForConstraints[7]*x[7]+coeffListForConstraints[8]*x[8]+coeffListForConstraints[9]*x[9]+coeffListForConstraints[10]*x[10]  >= rightHandSide )
    elseif operatorChoice == 3
        @constraint(model, con, coeffListForConstraints[1]*x[1]+coeffListForConstraints[2]*x[2]+coeffListForConstraints[3]*x[3]+coeffListForConstraints[4]*x[4]+coeffListForConstraints[5]*x[5]+coeffListForConstraints[6]*x[6]+coeffListForConstraints[7]*x[7]+coeffListForConstraints[8]*x[8]+coeffListForConstraints[9]*x[9]+coeffListForConstraints[10]*x[10]  <= rightHandSide )
    end
    
        

    coeffListForConstraints2 = rand((0:100), 9)
    rightHandSide2 = rand(0:1000)
    operatorChoice2 = rand(1:3)
    if operatorChoice2 == 1
        @constraint(model, con2, coeffListForConstraints2[1]*x[1]+coeffListForConstraints2[2]*x[2]+coeffListForConstraints2[3]*x[3]+coeffListForConstraints2[4]*x[4]+coeffListForConstraints[5]*x[5]+coeffListForConstraints[6]*x[6]+coeffListForConstraints[7]*x[7]+coeffListForConstraints[8]*x[8]+coeffListForConstraints[9]*x[9] == rightHandSide2 )
    elseif operatorChoice2 == 2
        @constraint(model, con2, coeffListForConstraints2[1]*x[1]+coeffListForConstraints2[2]*x[2]+coeffListForConstraints2[3]*x[3]+coeffListForConstraints2[4]*x[4]+coeffListForConstraints[5]*x[5]+coeffListForConstraints[6]*x[6]+coeffListForConstraints[7]*x[7]+coeffListForConstraints[8]*x[8]+coeffListForConstraints[9]*x[9] >= rightHandSide2 ) 
    elseif operatorChoice2 == 3
        @constraint(model, con2, coeffListForConstraints2[1]*x[1]+coeffListForConstraints2[2]*x[2]+coeffListForConstraints2[3]*x[3]+coeffListForConstraints2[4]*x[4]+coeffListForConstraints[5]*x[5]+coeffListForConstraints[6]*x[6]+coeffListForConstraints[7]*x[7]+coeffListForConstraints[8]*x[8]+coeffListForConstraints[9]*x[9] <= rightHandSide2 )  
    end

    coeffListForConstraints3 = rand((0:100), 8)
    rightHandSide3 = rand(0:1000)
    operatorChoice3 = rand(1:3)
    if operatorChoice3 == 1
        @constraint(model, con3, coeffListForConstraints3[1]*x[1]+coeffListForConstraints3[2]*x[2]+coeffListForConstraints3[3]*x[3]+coeffListForConstraints2[4]*x[4]+coeffListForConstraints[5]*x[5]+coeffListForConstraints[6]*x[6]+coeffListForConstraints[7]*x[7]+coeffListForConstraints[8]*x[8] == rightHandSide3 )
    elseif operatorChoice3 == 2
        @constraint(model, con3, coeffListForConstraints3[1]*x[1]+coeffListForConstraints3[2]*x[2]+coeffListForConstraints3[3]*x[3]+coeffListForConstraints2[4]*x[4]+coeffListForConstraints[5]*x[5]+coeffListForConstraints[6]*x[6]+coeffListForConstraints[7]*x[7]+coeffListForConstraints[8]*x[8] >= rightHandSide3 )    
    elseif operatorChoice3 == 3
        @constraint(model, con3, coeffListForConstraints3[1]*x[1]+coeffListForConstraints3[2]*x[2]+coeffListForConstraints3[3]*x[3]+coeffListForConstraints2[4]*x[4]+coeffListForConstraints[5]*x[5]+coeffListForConstraints[6]*x[6]+coeffListForConstraints[7]*x[7]+coeffListForConstraints[8]*x[8]<= rightHandSide3 )   
    end

    
        
    
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        timeResults = zeros(3)
        for k in 1:100
            set_optimizer_attributes(model,"Method" => 1)
            optimize!(model)
            timeResults[1] = timeResults[1] + solve_time(model)

            set_optimizer_attributes(model,"Method" => 2)
            optimize!(model)
            timeResults[2] = timeResults[2] + solve_time(model)

            set_optimizer_attributes(model,"Method" => 3)
            optimize!(model)
            timeResults[3] = timeResults[3] + solve_time(model)
        end

        global averageSolveTime
        solveTime = min(timeResults[1],timeResults[2],timeResults[3])/100
        averageSolveTime = averageSolveTime + solveTime
        label = argmin(timeResults)

        cd("/home/melo/Desktop/BITIRME")
        arrCurrent = zeros(58)
        arrCurrent[57] = label
        arrCurrent[58] = solveTime
        for i in 1:10 
            arrCurrent[i] = coeffList[i]    
        end
        for i in 15:24 
            arrCurrent[i] = coeffListForConstraints[i-14] 
        end
        if operatorChoice == 2
            arrCurrent[25] = -1
        elseif operatorChoice == 3
            arrCurrent[25] = 1
        end
        arrCurrent[28] = rightHandSide
            
        for i in 29:37 
            arrCurrent[i] = coeffListForConstraints2[i-28] 
        end
        if operatorChoice2 == 2
            arrCurrent[40] = -1
        elseif operatorChoice2 == 3
            arrCurrent[40] = 1
        end
        arrCurrent[42] = rightHandSide2

        for i in 43:50 
            arrCurrent[i] = coeffListForConstraints3[i-42] 
        end
        if operatorChoice3 == 2
            arrCurrent[55] = -1
        elseif operatorChoice3 == 3
            arrCurrent[55] = 1
        end
        arrCurrent[56] = rightHandSide3
        for k in 1:58
            arr[numberOfData,k] = arrCurrent[k]
        end
        
    else
        generateProblem(numberOfData)
    end
    
end

averageSolveTime = 0
arr = zeros(10000,58)      # first parameter -> how many samples
for k in 1:10000
    generateProblem(k)
end
averageSolveTime = averageSolveTime / 10000
println(averageSolveTime)

df = convert(DataFrame,arr)
CSV.write(string("/home/melo/Desktop/BITIRME/allData.csv"), df)        #to create train file
#CSV.write(string("/home/melo/Desktop/BITIRME/testData.csv"), df)      #to create test file



