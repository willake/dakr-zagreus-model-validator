local helper = {}
-- Function to calculate Mean Absolute Error (MAE)
function helper.meanAbsoluteError(predicted, actual)
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

-- Shuffle the dataset
function helper.shuffleDataset(dataset)
    for i = #dataset, 1, -1 do
        local r = math.random(1, i)
        local temp = dataset[r]
        dataset[r] = dataset[i]
        dataset[i] = temp
    end
end

function helper.splitDatasetToK(dataset, k)
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

	for i = 3, #fileLines do
		if i%2 == 1 then
			local tempInputs = {}
			for input in fileLines[i]:gmatch("%S+") do table.insert(tempInputs, tonumber(input)) end
			local tempOutputs = {}
			for output in fileLines[i+1]:gmatch("%S+") do table.insert(tempOutputs, tonumber(output)) end
			table.insert(trainingData, {tempInputs, tempOutputs})
		end
	end
	return(trainingData)
end


return helper