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

helper.shuffleDataset(dataset)

local counts = helper.countActions(dataset)

print(table.concat(counts, ", "))

local folds = helper.splitDatasetToKFolds(dataset, k) -- for k-fold cross validation

local gTrainingACCSum = 0 -- Whether the predicted action holds highest probability is the same as ground truth
local gTrainingATCSum = 0 -- MSE of charge time

local gTestingACCSum = 0 -- Whether the predicted action holds highest probability is the same as ground truth
local gTestingATCSum = 0 -- MSE of charge time

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

    -- AE = Action error, ACC = Action correct count, ATC = Action total count, CTE = Charge time error
    -- get training error
    -- validate model using folds other than i-th fold, which is the training set
    local trainingAESum = 0
    local trainingACCSum = 0
    local trainingATCSum = 0
    for idx = 1, k do
        if idx ~= testIdx then
            local localErrors = helper.calculateErrorOfFold(network, folds[idx])
            trainingAESum = trainingAESum + localErrors[1]
            trainingACCSum = trainingACCSum + localErrors[2]
            trainingATCSum = trainingATCSum + localErrors[3]
        end
    end

    local trainingAE = trainingAESum / (k - 1)

    -- get testing error
    -- validate model using i-th fold, which is the testing set
    local testingErrors = helper.calculateErrorOfFoldV5(network, folds[testIdx])
    local testingAE = testingErrors[1]
    local testingACC = testingErrors[2]
    local testingATC = testingErrors[3]
    local testingCTE = testingErrors[4]

    print(string.format("%dth iteration - training time: %.2f", 
        testIdx, time))
    print(string.format("training action correctness: %.2f(%d/%d)",
        trainingACCSum / trainingATCSum, trainingACCSum, trainingATCSum))
    print(string.format("testing action correctness: %.2f(%d/%d)", 
        testingACC / testingATC, testingACC, testingATC, testingCTE))

    gTrainingACCSum = gTrainingACCSum + trainingACCSum
    gTrainingATCSum = gTrainingATCSum + trainingATCSum

    gTestingACCSum = gTestingACCSum + testingACC
    gTestingATCSum = gTestingATCSum + testingATC
end 

print("Cross-validation result -")
print(string.format("training action correctness: %.2f", 
    gTrainingACCSum / gTrainingATCSum))
print(string.format("testing action correctness: %.2f", 
    gTestingACCSum / gTestingATCSum))