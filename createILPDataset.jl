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
    @variable(model,0 <= x[1:5] <= 100)
    coeffList = rand((0:100), 5)
    @objective(model, Max, coeffList[1]*x[1]+coeffList[2]*x[2]+coeffList[3]*x[3]+coeffList[4]*x[4]+coeffList[5]*x[5])
    feasibleFound = 0

    coeffListForConstraints = rand((0:100), 5)
    rightHandSide = rand(0:1000)
    operatorChoice = rand(1:3)
    if operatorChoice == 1
        @constraint(model, con, coeffListForConstraints[1]*x[1]+coeffListForConstraints[2]*x[2]+coeffListForConstraints[3]*x[3]+coeffListForConstraints[4]*x[4]+coeffListForConstraints[5]*x[5] == rightHandSide )
    elseif operatorChoice == 2
        @constraint(model, con, coeffListForConstraints[1]*x[1]+coeffListForConstraints[2]*x[2]+coeffListForConstraints[3]*x[3]+coeffListForConstraints[4]*x[4]+coeffListForConstraints[5]*x[5] >= rightHandSide )
    elseif operatorChoice == 3
        @constraint(model, con, coeffListForConstraints[1]*x[1]+coeffListForConstraints[2]*x[2]+coeffListForConstraints[3]*x[3]+coeffListForConstraints[4]*x[4]+coeffListForConstraints[5]*x[5] <= rightHandSide )
    end
    
        

    coeffListForConstraints2 = rand((0:100), 4)
    rightHandSide2 = rand(0:1000)
    operatorChoice2 = rand(1:3)
    if operatorChoice2 == 1
        @constraint(model, con2, coeffListForConstraints2[1]*x[1]+coeffListForConstraints2[2]*x[2]+coeffListForConstraints2[3]*x[3]+coeffListForConstraints2[4]*x[4] == rightHandSide2 )
    elseif operatorChoice2 == 2
        @constraint(model, con2, coeffListForConstraints2[1]*x[1]+coeffListForConstraints2[2]*x[2]+coeffListForConstraints2[3]*x[3]+coeffListForConstraints2[4]*x[4] >= rightHandSide2 ) 
    elseif operatorChoice2 == 3
        @constraint(model, con2, coeffListForConstraints2[1]*x[1]+coeffListForConstraints2[2]*x[2]+coeffListForConstraints2[3]*x[3]+coeffListForConstraints2[4]*x[4] <= rightHandSide2 )  
    end

    coeffListForConstraints3 = rand((0:100), 3)
    rightHandSide3 = rand(0:1000)
    operatorChoice3 = rand(1:3)
    if operatorChoice3 == 1
        @constraint(model, con3, coeffListForConstraints3[1]*x[1]+coeffListForConstraints3[2]*x[2]+coeffListForConstraints3[3]*x[3] == rightHandSide3 )
    elseif operatorChoice3 == 2
        @constraint(model, con3, coeffListForConstraints3[1]*x[1]+coeffListForConstraints3[2]*x[2]+coeffListForConstraints3[3]*x[3] >= rightHandSide3 )    
    elseif operatorChoice3 == 3
        @constraint(model, con3, coeffListForConstraints3[1]*x[1]+coeffListForConstraints3[2]*x[2]+coeffListForConstraints3[3]*x[3] <= rightHandSide3 )   
    end

    
        
    
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        timeResults = zeros(3)
        for k in 1:10
            set_optimizer_attributes(model,"Method" => 1)
            optimize!(model)
            timeResults[1] = timeResults[1] + solve_time(model)

            set_optimizer_attributes(model,"Method" => 3)
            optimize!(model)
            timeResults[2] = timeResults[2] + solve_time(model)

            set_optimizer_attributes(model,"Method" => 5)
            optimize!(model)
            timeResults[3] = timeResults[3] + solve_time(model)
        end

        label = argmin(timeResults)

        arrCurrent = zeros(37)
        arrCurrent[37] = label
        for i in 1:5 
            arrCurrent[i] = coeffList[i]    
        end
        for i in 10:14 
            arrCurrent[i] = coeffListForConstraints[i-9] 
        end
        if operatorChoice == 2
            arrCurrent[15] = -1
        elseif operatorChoice == 3
            arrCurrent[15] = 1
        end
        arrCurrent[18] = rightHandSide
            
        for i in 19:22 
            arrCurrent[i] = coeffListForConstraints2[i-18] 
        end
        if operatorChoice2 == 2
            arrCurrent[25] = -1
        elseif operatorChoice2 == 3
            arrCurrent[25] = 1
        end
        arrCurrent[27] = rightHandSide2

        for i in 28:30 
            arrCurrent[i] = coeffListForConstraints3[i-27] 
        end
        if operatorChoice3 == 2
            arrCurrent[35] = -1
        elseif operatorChoice3 == 3
            arrCurrent[35] = 1
        end
        arrCurrent[36] = rightHandSide3
        for k in 1:37
            arr[numberOfData,k] = arrCurrent[k]
        end
        
    else
        generateProblem(numberOfData)
    end
    
end


arr = zeros(60000,37)      # first parameter -> how many samples
for k in 1:60000
    generateProblem(k)
end

df = convert(DataFrame,arr)
CSV.write(string("allData.csv"), df)        #to create train file
#CSV.write(string("testData.csv"), df)      #to create test file



