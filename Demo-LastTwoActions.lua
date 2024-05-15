local luann = require("luann")
local helper = require("helper")
math.randomseed(89890)

local learningRate = 5 -- set between 1, 100
local attempts = 10 -- number of times to do backpropagation
local threshold = 1 -- steepness of the sigmoid curve

--create a network with 2 inputs, 3 hidden cells, and 1 output
-- myNetwork = luann:new({6, 6, 6, 4}, learningRate, threshold)
local myNetwork = luann:new({10, 10, 10, 4}, learningRate, threshold)

local trainingData = helper.loadTrainingDataFromFile("DZrecord.log")
local testingData = helper.loadTrainingDataFromFile("DZtest.log")

-- make new training data
for i = 3, #trainingData do
    local prev2 = trainingData[i - 2][2]

    for j = 1, 4 do
        table.insert(trainingData[i][1], prev2[j])
    end

    print(table.concat(trainingData[i][1], ", "))

    -- print(#trainingData[i][1])
end

-- remove the first two since they don't have second-previous data
table.remove(trainingData, 1)
table.remove(trainingData, 1)

-- helper.shuffleDataset(trainingData)

for i = 3, #testingData do
    local prev2 = testingData[i - 2][2]

    for j = 1, 4 do
        table.insert(testingData[i][1], prev2[j])
    end
end

local x = os.clock()

--run backpropagation (bp)
for i = 1, attempts do
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

local mae = helper.meanAbsoluteError(predicted, actual)
print(mae)