local luann = require("luann")
math.randomseed(89890)

-- Function to calculate Mean Absolute Error (MAE)
local function meanAbsoluteError(predicted, actual)
    local sumError = 0
    local n = #predicted
    local m = #predicted[1] -- num of outputs

    for i = 1, n do
        local outputError = 0
        for j = 1, m do
            outputError = outputError + math.abs(predicted[i][j] - actual[i][j])
        end
        local mean = outputError / m
        sumError = sumError + mean
    end

    return sumError / n
end

learningRate = 50 -- set between 1, 100
attempts = 10 -- number of times to do backpropagation
threshold = 1 -- steepness of the sigmoid curve

--create a network with 2 inputs, 3 hidden cells, and 1 output
myNetwork = luann:new({6, 6, 6, 4}, learningRate, threshold)

local trainingData = luann:loadTrainingDataFromFile("DZrecord.log")
local testingData = luann:loadTrainingDataFromFile("DZtest.log")

local x = os.clock()

--run backpropagation (bp)
for i = 1,attempts do
    for j = 1, #trainingData do
        myNetwork:bp(trainingData[j][1], trainingData[j][2])    
    end
end

print(string.format("training time: %.2f\n", os.clock() - x))

local predicted = {}
local actual = {}
for i = 1, #testingData do
    myNetwork:activate(testingData[i][1])
    local output = {
        myNetwork[4].cells[1].signal, myNetwork[4].cells[2].signal, 
        myNetwork[4].cells[3].signal, myNetwork[4].cells[4].signal}
    table.insert(predicted, output)
    table.insert(actual, testingData[i][2])
end

local mae = meanAbsoluteError(predicted, actual)
print(mae)