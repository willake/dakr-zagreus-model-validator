local helper = {}
-- Function to calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
function helper.calculateAbsoluteError(prediction, truth, outputCount)
    local sumError = 0
    for i = 1, outputCount do -- only test first three outputs
        sumError = sumError + math.abs(prediction[i] - truth[i])
    end

    return sumError / outputCount
end

function helper.calculateSquaredError(prediction, truth, outputCount)
    local sumError = 0
    for i = 1, outputCount do -- only test first three outputs
        local diff = truth[i] - prediction[i]
        sumError = sumError + (diff * diff)
    end

    return sumError / outputCount
end

-- Custom metric, check if the action holds highest probability is the same as ground truth 
function helper.calcuateActionCorrectness(prediction, truth, actionCount)
    -- action index of truth, 0 = dash toward, 1 = attack, 2 = special attack, 3 = dash away
    local p = helper.getHighestAction(prediction)
    local t = helper.getHighestAction(truth)

    if p == t then
        return 1
    end

    return 0
end

-- Shuffle the dataset
function helper.shuffleDataset(dataset)
    for i = #dataset, 1, -1 do
        local r = math.random(1, i)
        local temp = dataset[r]
        dataset[r] = dataset[i]
        dataset[i] = temp
    end
end

function helper.splitDatasetToKFolds(dataset, k)
    local n = #dataset
    local partSize = math.floor(n / k)
    local remainder = n % k
    local startIndex = 1
    local splited = {}

    for i = 1, k do
        local part = {}
        local size = partSize
        if i <= remainder then
            size = size + 1
        end

        for j = 1, size do
            table.insert(part, dataset[startIndex])
            startIndex = startIndex + 1
        end

        table.insert(splited, part)
    end

    return splited
end

function helper.loadDatasetFromFile(fileName)
	local trainingData = {}
	local fileLines = {}
    local f = io.open(fileName, "rb")
		 for line in f:lines() do
			table.insert (fileLines, line);
		 end
	f:close()

	for i = 4, #fileLines do
		if i%2 == 0 then
			local tempInputs = {}
			for input in fileLines[i]:gmatch("%S+") do table.insert(tempInputs, tonumber(input)) end
			local tempOutputs = {}
			for output in fileLines[i+1]:gmatch("%S+") do table.insert(tempOutputs, tonumber(output)) end
			table.insert(trainingData, {tempInputs, tempOutputs})
		end
	end
	return(trainingData)
end

function helper.calculateErrorOfFoldV4(network, fold)
    local predActionCounts = {0, 0, 0, 0, 0, 0}
    local truthActionCounts = {0, 0, 0, 0, 0, 0}
    local actionCorrectnessSum = 0
    local chargeTimeSum = 0

    for i = 1, #fold do
        network:activate(fold[i][1]) 
        local prediction = {
            network[4].cells[1].signal, network[4].cells[2].signal, 
            network[4].cells[3].signal, network[4].cells[4].signal, network[4].cells[5].signal
        }
        print(table.concat(prediction, ", "))
        local p = helper.getHighestAction(prediction)
        local t = helper.getHighestAction(fold[i][2])
        predActionCounts[p] = predActionCounts[p] + 1
        truthActionCounts[t] = truthActionCounts[t] + 1
        actionCorrectnessSum = actionCorrectnessSum + helper.calcuateActionCorrectness(prediction, fold[i][2], 4)
        chargeTimeSum = chargeTimeSum + helper.calculateSquaredError({ prediction[5] }, { fold[i][2][5] }, 1) 
    end

    local chargeTimeError = chargeTimeSum / #fold

    return {
        actionCorrectness = actionCorrectnessSum,
        actionNum = #fold, 
        predActionCounts = predActionCounts, 
        truthActionCounts = truthActionCounts,
        chargeTimeError = chargeTimeError
    }
end


function helper.calculateErrorOfFoldV5(network, fold)
    local predActionCounts = {0, 0, 0, 0, 0, 0}
    local truthActionCounts = {0, 0, 0, 0, 0, 0}
    local actionCorrectnessSum = 0

    for i = 1, #fold do
        network:activate(fold[i][1]) 
        local prediction = {
            network[4].cells[1].signal, network[4].cells[2].signal, 
            network[4].cells[3].signal, network[4].cells[4].signal, network[4].cells[5].signal
        }
        print(table.concat(prediction, ", "))
        local p = helper.getHighestAction(prediction)
        local t = helper.getHighestAction(fold[i][2])
        predActionCounts[p] = predActionCounts[p] + 1
        truthActionCounts[t] = truthActionCounts[t] + 1
        actionCorrectnessSum = actionCorrectnessSum + helper.calcuateActionCorrectness(prediction, fold[i][2], 5)
    end

    return {
        actionCorrectness = actionCorrectnessSum,
        actionNum = #fold, 
        predActionCounts = predActionCounts, 
        truthActionCounts = truthActionCounts,
    }
end

function helper.calculateErrorOfFoldV6(network, fold)
    local predActionCounts = {0, 0, 0, 0, 0, 0}
    local truthActionCounts = {0, 0, 0, 0, 0, 0}
    local actionCorrectnessSum = 0

    for i = 1, #fold do
        network:activate(fold[i][1]) 
        local prediction = {
            network[4].cells[1].signal, network[4].cells[2].signal, 
            network[4].cells[3].signal, network[4].cells[4].signal, network[4].cells[5].signal, network[4].cells[6].signal
        }
        print(table.concat(prediction, ", "))
        local p = helper.getHighestAction(prediction)
        local t = helper.getHighestAction(fold[i][2])
        predActionCounts[p] = predActionCounts[p] + 1
        truthActionCounts[t] = truthActionCounts[t] + 1
        actionCorrectnessSum = actionCorrectnessSum + helper.calcuateActionCorrectness(prediction, fold[i][2], 6)
    end

    return {
        actionCorrectness = actionCorrectnessSum,
        actionNum = #fold, 
        predActionCounts = predActionCounts, 
        truthActionCounts = truthActionCounts,
    }
end

-- Count the amounth of each actions within the dataset
function helper.countActions(dataset)
    local template = dataset[1][2]
    local counts = {}
    for i = 1, #template do
        table.insert(counts, 0)
    end

    for i = 1, #dataset do
        local action = dataset[i][2]
        local b = 1 -- biggest
        local bp = 0.0
        for j = 1, #action do
            if bp < action[j] then
                b = j
                bp = action[j]
            end
        end
        counts[b] = counts[b] + 1 
    end

    return counts
end

-- Get highest action index within a list of action probability
function helper.getHighestAction(actionProbs)
    local p = 1
    local pm = 0 -- prob max
    for i = 1, #actionProbs do
        if actionProbs[i] > pm then
            p = i
            pm = actionProbs[i]
        end
    end
    return p
end

return helper