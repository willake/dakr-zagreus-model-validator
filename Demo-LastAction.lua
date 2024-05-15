local luann = require("luann")
math.randomseed(89890)
local helper = require("helper")

local learningRate = 5 -- set between 1, 100
local attempts = 10 -- number of times to do backpropagation
local threshold = 1 -- steepness of the sigmoid curve

--create a network with 6 inputs, 6 hidden cells, and 4 output
myNetwork = luann:new({6, 6, 6, 4}, learningRate, threshold)

local trainingData = helper.loadDatasetFromFile("DZrecord.log")
local testingData = helper.loadDatasetFromFile("DZtest.log")

helper.shuffleDataset(trainingData)

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

local mae = helper.meanAbsoluteError(predicted, actual)
print(mae)