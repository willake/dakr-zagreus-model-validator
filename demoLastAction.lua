local luann = require("luann")
math.randomseed(89890)
local helper = require("helper")

local learningRate = 1 -- set between 1, 100
local epoch = 5 -- number of times to do backpropagation
local threshold = 1 -- steepness of the sigmoid curve

local k = 5
local dataset = helper.loadDatasetFromFile("DZrecord.log")
local folds = helper.splitDatasetToKFolds(dataset, k) -- for k-fold cross validation

helper.shuffleDataset(dataset)

--run backpropagation (bp)
local globalErrorSum = 0 -- MSE of actions probability
local globalActionCorrectnessSum = 0 -- Whether the predicted action holds highest probability is the same as ground truth
local globalChargeTimeErrorSum = 0 -- MSE of charge time
for testIdx = 1, k do -- do k times
    local network = luann:new({6, 6, 6, 4}, learningRate, threshold)

    local start = os.clock()
    for _ = 1, epoch do
        for i = 1, k do -- run through all folds
            if testIdx ~= i then -- if the index is not the test set
                for j = 1, #folds[i] do
                    network:bp(folds[i][j][1], folds[i][j][2]) 
                end
            end
        end
    end

    local time = os.clock() - start

    -- validate model using i-th fold, which is the training set
    local testset = folds[testIdx] 
    local localErrorSum = 0
    local localActionCorrectnessSum = 0
    local localChargeTimeErrorSum = 0
    
    for i = 1, #testset do
        network:activate(testset[i][1]) 
        local prediction = {
            network[4].cells[1].signal, network[4].cells[2].signal, 
            network[4].cells[3].signal, network[4].cells[4].signal
        }
        localErrorSum = localErrorSum + helper.calculateSquaredError(prediction, testset[i][2], 3)
        localActionCorrectnessSum = localActionCorrectnessSum + helper.calcuateActionCorrectness(prediction, testset[i][2], 3)
        localChargeTimeErrorSum = localChargeTimeErrorSum 
            + helper.calculateSquaredError({ prediction[4] }, { testset[i][2][4] }, 1) -- 4th element is charge time
    end

    local localError = localErrorSum / #testset
    local localActionCorrectness = localActionCorrectnessSum / #testset
    local localChargeTimeError = localChargeTimeErrorSum / #testset

    print(string.format("%dth iteration - training time: %.2f, action correctness: %.2f(%d/%d), charge time error: %.3f", 
        testIdx, time, localActionCorrectness, localActionCorrectnessSum, #testset, localChargeTimeError))

    globalErrorSum = globalErrorSum + localError
    globalActionCorrectnessSum = globalActionCorrectnessSum + localActionCorrectness
    globalChargeTimeErrorSum = globalChargeTimeErrorSum + localChargeTimeError
end 

local error = globalErrorSum / k
local actionCorrectness = globalActionCorrectnessSum / k
local chargeTimeError = globalChargeTimeErrorSum / k 

print(string.format("Cross-validation result - MSE: %.3f, action correctness: %.2f, charge time error: %.3f", 
    error, actionCorrectness, chargeTimeError))