local luann = require("luann")
math.randomseed(89890)
local helper = require("helper")

local learningRate = 5 -- set between 1, 100
local epoch = 3 -- number of times to do backpropagation
local threshold = 1 -- steepness of the sigmoid curve

local k = 5
local dataset = helper.loadDatasetFromFile("DZrecord.log")

print(string.format("Data count: %d", #dataset))

-- make new training data which attach second-last action to the current state
for i = 4, #dataset do
    local prev = dataset[i - 1][2]
    local prev2 = dataset[i - 2][2]

    for j = 1, 4 do
        table.insert(dataset[i][1], prev[j])
    end

    for j = 1, 4 do
        table.insert(dataset[i][1], prev2[j])
    end
end

-- remove the first since it doesn't have second-previous data
table.remove(dataset, 1)

helper.shuffleDataset(dataset)

local folds = helper.splitDatasetToKFolds(dataset, k) -- for k-fold cross validation

local gTrainingAESum = 0 -- MSE of actions probability
local gTrainingACCSum = 0 -- Whether the predicted action holds highest probability is the same as ground truth
local gTrainingATCSum = 0 -- MSE of charge time
local gTrainingCTESum = 0

local gTestingAESum = 0 -- MSE of actions probability
local gTestingACCSum = 0 -- Whether the predicted action holds highest probability is the same as ground truth
local gTestingATCSum = 0 -- MSE of charge time
local gTestingCTESum = 0
for testIdx = 1, k do -- do k times
    local network = luann:new({11, 9, 9, 4}, learningRate, threshold)

    local start = os.clock()
    for _ = 1, epoch do
        for i = 1, k do -- run through all folds
            if testIdx ~= i then -- if the index is not the test set, train model
                for j = 1, #folds[i] do
                    network:bp(folds[i][j][1], folds[i][j][2]) 
                end
            end
        end
    end

    local time = os.clock() - start

    -- AE = Action error, ACC = Action correct count, ATC = Action total count, CTE = Charge time error
    -- get training error
    -- validate model using folds other than i-th fold, which is the training set
    local trainingAESum = 0
    local trainingACCSum = 0
    local trainingATCSum = 0
    local trainingCTESum = 0
    for idx = 1, k do
        if idx ~= testIdx then
            local localErrors = helper.calculateErrorOfFold(network, folds[idx])
            trainingAESum = trainingAESum + localErrors[1]
            trainingACCSum = trainingACCSum + localErrors[2]
            trainingATCSum = trainingATCSum + localErrors[3]
            trainingCTESum = trainingCTESum + localErrors[4]
        end
    end

    local trainingAE = trainingAESum / (k - 1)
    local trainingCTE = trainingCTESum / (k - 1)

    -- get testing error
    -- validate model using i-th fold, which is the testing set
    local testingErrors = helper.calculateErrorOfFold(network, folds[testIdx])
    local testingAE = testingErrors[1]
    local testingACC = testingErrors[2]
    local testingATC = testingErrors[3]
    local testingCTE = testingErrors[4]

    print(string.format("%dth iteration - training time: %.2f", 
        testIdx, time))
    print(string.format("training action correctness: %.2f(%d/%d), training charge time error: %.3f", 
        trainingACCSum / trainingATCSum, trainingACCSum, trainingATCSum, trainingCTE))
    print(string.format("testing action correctness: %.2f(%d/%d), testing charge time error: %.3f", 
        testingACC / testingATC, testingACC, testingATC, testingCTE))

    gTrainingACCSum = gTrainingACCSum + trainingACCSum
    gTrainingATCSum = gTrainingATCSum + trainingATCSum
    gTrainingCTESum = gTrainingCTESum + trainingCTE  

    gTestingACCSum = gTestingACCSum + testingACC
    gTestingATCSum = gTestingATCSum + testingATC
    gTestingCTESum = gTestingCTESum + testingCTE
end 

print("Cross-validation result -")
print(string.format("training action correctness: %.2f, training charge time MSE: %.3f", 
    gTrainingACCSum / gTrainingATCSum, gTrainingCTESum / k))
print(string.format("testing action correctness: %.2f, testing charge time MSE: %.3f", 
    gTestingACCSum / gTestingATCSum, gTestingCTESum / k))