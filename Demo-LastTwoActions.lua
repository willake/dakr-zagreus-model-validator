local luann = require("luann")
math.randomseed(89890)

-- Function to calculate Mean Absolute Error (MAE)
local function meanAbsoluteError(predicted, actual)
    local sumError = 0
    local n = #predicted
    local m = #predicted[1] -- num of outputs

    for i = 1, n do
        local outputError = 0
        for j = 1, 3 do -- only test first three outputs
            outputError = outputError + math.abs(predicted[i][j] - actual[i][j])
        end
        local mean = outputError / m
        sumError = sumError + mean
    end

    return sumError / n
end

local learningRate = 5 -- set between 1, 100
local attempts = 10 -- number of times to do backpropagation
local threshold = 1 -- steepness of the sigmoid curve

--create a network with 2 inputs, 3 hidden cells, and 1 output
-- myNetwork = luann:new({6, 6, 6, 4}, learningRate, threshold)
local myNetwork = luann:new({10, 10, 10, 4}, learningRate, threshold)

local trainingData = luann:loadTrainingDataFromFile("DZrecord.log")
local testingData = luann:loadTrainingDataFromFile("DZtest.log")

-- make new training data
for i = 3, #trainingData do
    local prev2 = trainingData[i - 2][2]

    for j = 1, 4 do
        table.insert(trainingData[i][1], prev2[j])
    end

    -- print(#trainingData[i][1])
end

for i = 3, #testingData do
    local prev2 = testingData[i - 2][2]

    for j = 1, 4 do
        table.insert(testingData[i][1], prev2[j])
    end
end

local x = os.clock()

--run backpropagation (bp)
for i = 1,attempts do
    for j = 3, #trainingData do
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