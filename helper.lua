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
    -- action index of truth, 0 = dash, 1 = attack, 2 = special attack
    local p = 0
    local pm = 0 -- max of p
    local t = 0
    local tm = 0 -- max of t
    for i = 1, actionCount do -- only test first three outputs
        if prediction[i] > pm then
            p = i
            pm = prediction[i]
        end

        if truth[i] > tm then
            t = i
            tm = truth[i]
        end
    end

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

function helper.calculateErrorOfFold(network, fold)
    local actionErrorSum = 0
    local actionCorrectnessSum = 0
    local chargeTimeErrorSum = 0

    for i = 1, #fold do
        network:activate(fold[i][1]) 
        local prediction = {
            network[4].cells[1].signal, network[4].cells[2].signal, 
            network[4].cells[3].signal, network[4].cells[4].signal
        }
        -- print(table.concat(prediction, ", "))
        actionErrorSum = actionErrorSum + helper.calculateSquaredError(prediction, fold[i][2], 3)
        actionCorrectnessSum = actionCorrectnessSum + helper.calcuateActionCorrectness(prediction, fold[i][2], 3)
        chargeTimeErrorSum = chargeTimeErrorSum 
            + helper.calculateSquaredError({ prediction[4] }, { fold[i][2][4] }, 1) -- 4th element is charge time
    end

    local actionError = actionErrorSum / #fold
    local chargeTimeError = chargeTimeErrorSum / #fold

    return {actionError, actionCorrectnessSum, #fold, chargeTimeError}
end


return helper