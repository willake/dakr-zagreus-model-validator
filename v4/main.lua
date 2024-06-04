package.path = package.path .. ';../?.lua;../?.lua;../?.lua;target/?.lua'

local luann = require("luann")
math.randomseed(89890)
local helper = require("helper")

local learningRate = 3 -- set between 1, 100
local epoch = 10 -- number of times to do backpropagation
local threshold = 1 -- steepness of the sigmoid curve

local k = 5
local dataset = helper.loadDatasetFromFile("DZrecord.log")

local consideration = 2 -- conside how many past actions, 2 means 2 past actions
-- insert last actions to each data, this provides more information for training the model
for i = 1 + consideration, #dataset do
    for j = 1, consideration do
        local prev = dataset[i - j][2]    
        
        for k = 1, #prev do
            table.insert(dataset[i][1], prev[k])
        end
    end
end

-- remove the first since it doesn't have second-previous data
table.remove(dataset, 1)

-- shuffle the data set, improving generalizability
helper.shuffleDataset(dataset)

-- count the amount of each actions
local counts = helper.countActions(dataset)

print(table.concat(counts, ", "))

local folds = helper.splitDatasetToKFolds(dataset, k) -- for k-fold cross validation

local gTrainingCAC = 0 -- global training correct actions count
local gTrainingTAC = 0 -- global training total actions count
local gTrainingCTE = 0 -- global charge time error

local gTestingCAC = 0 -- global testing correct actions count
local gTestingTAC = 0 -- global testing total actions count
local gTestingCTE = 0 -- global testing charge time error

for testIdx = 1, k do -- do k times
    local network = luann:new({16, 11, 11, 5}, learningRate, threshold)

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

    -- CAC = Correct actions count, TAC = Total actions count
    -- get training error
    -- validate model using folds other than i-th fold, which is the training set
    local trainingCAC = 0
    local trainingTAC = 0
    local trainingPredActions = {0, 0, 0, 0, 0, 0}
    local trainingTruthActions = {0, 0, 0, 0, 0, 0}
    local trainingCTE = 0

    for idx = 1, k do
        if idx ~= testIdx then
            local localResult = helper.calculateErrorOfFoldV4(network, folds[idx])
            trainingCAC = trainingCAC + localResult.actionCorrectness
            trainingTAC = trainingTAC + localResult.actionNum
            for i = 1, #localResult.predActionCounts do
                trainingPredActions[i] = trainingPredActions[i] + localResult.predActionCounts[i]
            end
            for i = 1, #localResult.truthActionCounts do
                trainingTruthActions[i] = trainingTruthActions[i] + localResult.truthActionCounts[i]
            end
            trainingCTE = trainingCTE + localResult.chargeTimeError
        end
    end
    -- get testing error
    -- validate model using i-th fold, which is the testing set
    local result = helper.calculateErrorOfFoldV4(network, folds[testIdx])
    local testingCAC = result.actionCorrectness
    local testingTAC = result.actionNum
    local testingPredActions = result.predActionCounts
    local testingTruthActions = result.truthActionCounts
    local testingCTE = result.chargeTimeError

    print(string.format("%dth iteration - training time: %.2f", 
        testIdx, time))
    print(string.format("predict Actions: %s", table.concat(trainingPredActions, ", ")))
    print(string.format("truth Actions: %s", table.concat(trainingTruthActions, ", ")))
    print(string.format("training action correctness: %.2f(%d/%d), chargeTimeError: %.3f",
        trainingCAC / trainingTAC, trainingCAC, trainingTAC, trainingCTE / (k - 1)))
    print(string.format("predict Actions: %s", table.concat(testingPredActions, ", ")))
    print(string.format("truth Actions: %s", table.concat(testingTruthActions, ", ")))
    print(string.format("testing action correctness: %.2f(%d/%d) chargeTimeError: %.3f", 
        testingCAC / testingTAC, testingCAC, testingTAC, testingCTE))

    gTrainingCAC = gTrainingCAC + trainingCAC
    gTrainingTAC = gTrainingTAC + trainingTAC
    gTrainingCTE = gTrainingCTE + (trainingCTE / (k - 1))

    gTestingCAC = gTestingCAC + testingCAC
    gTestingTAC = gTestingTAC + testingTAC
    gTestingCTE = gTestingCTE + testingCTE
end 

print("Cross-validation result -")
print(string.format("training action correctness: %.2f, chatge time error: %.2f", 
    gTrainingCAC / gTrainingTAC, gTrainingCTE / k))
print(string.format("testing action correctness: %.2f", 
    gTestingCAC / gTestingTAC, gTestingCTE / k))