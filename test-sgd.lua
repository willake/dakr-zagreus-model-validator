-- Define sigmoid activation function
local function sigmoid(x)
    return 1.0 / (1.0 + math.exp(-x))
end

-- Define derivative of sigmoid activation function
local function sigmoidDerivative(x)
    local sigmoidX = sigmoid(x)
    return sigmoidX * (1 - sigmoidX)
end

-- Define Cell class
local Cell = {}

function Cell:new(inputSize)
    local cell = {
        weights = {},
        bias = math.random()
    }

    for i = 1, inputSize do
        cell.weights[i] = math.random()
    end

    setmetatable(cell, self)
    self.__index = self
    return cell
end

function Cell:activate(inputs)
    local weightedSum = self.bias
    for i, input in ipairs(inputs) do
        weightedSum = weightedSum + self.weights[i] * input
    end
    return sigmoid(weightedSum)
end

-- Define Layer class
local Layer = {}

function Layer:new(inputSize, numCells)
    local layer = {
        cells = {}
    }

    for i = 1, numCells do
        layer.cells[i] = Cell:new(inputSize)
    end

    setmetatable(layer, self)
    self.__index = self
    return layer
end

function Layer:activate(inputs)
    local outputs = {}
    for i, cell in ipairs(self.cells) do
        outputs[i] = cell:activate(inputs)
    end
    return outputs
end

-- Define neural network class
local NeuralNetwork = {}

function NeuralNetwork:new(inputNodes, hiddenNodesList, outputNodes)
    local nn = {
        inputNodes = inputNodes,
        hiddenLayers = {},
        outputLayer = Layer:new(hiddenNodesList[#hiddenNodesList], outputNodes)
    }

    -- Initialize hidden layers
    local inputSize = inputNodes
    for _, hiddenNodes in ipairs(hiddenNodesList) do
        table.insert(nn.hiddenLayers, Layer:new(inputSize, hiddenNodes))
        inputSize = hiddenNodes
    end

    setmetatable(nn, self)
    self.__index = self
    return nn
end

function NeuralNetwork:feedforward(inputs)
    local hiddenInputs = inputs

    -- Calculate hidden layers outputs
    local hiddenOutputs = {}
    for _, layer in ipairs(self.hiddenLayers) do
        hiddenOutputs[#hiddenOutputs + 1] = layer:activate(hiddenInputs)
        hiddenInputs = hiddenOutputs[#hiddenOutputs]
    end

    -- Calculate output layer outputs
    local outputs = self.outputLayer:activate(hiddenOutputs[#hiddenOutputs])

    return outputs, hiddenOutputs
end

function NeuralNetwork:backpropagate(inputs, targets)
    
end


function NeuralNetwork:updateWeights(inputs, targets, learning_rate)
    local outputs, hiddenOutputs = self:feedforward(inputs)

    -- Calculate output layer errors
    local outputErrors = {}
    for i = 1, #outputs do
        outputErrors[i] = targets[i] - outputs[i]
    end

    -- Update weights and biases for output layer
    for i, cell in ipairs(self.outputLayer.cells) do
        for j, output in ipairs(hiddenOutputs[#hiddenOutputs]) do
            cell.weights[j] = cell.weights[j] + learning_rate * outputErrors[i] * sigmoidDerivative(outputs[i]) * output
        end
        cell.bias = cell.bias + learning_rate * outputErrors[i] * sigmoidDerivative(outputs[i])
    end

    local hiddenErrors = {}

    -- Update weights and biases for hidden layers
    for k = #self.hiddenLayers, 1, -1 do
        local layer = self.hiddenLayers[k]
        local nextLayer = k == #self.hiddenLayers and self.outputLayer or self.hiddenLayers[k + 1]
        local nextLayerErrors = k == #self.hiddenLayers and outputErrors or hiddenErrors[k + 1]

        for i, cell in ipairs(layer.cells) do
            for j, next_cell in ipairs(nextLayer.cells) do
                local error = 0
                for l, next_error in ipairs(nextLayerErrors) do
                    error = error + next_error * next_cell.weights[i]
                end
                cell.weights[j] = cell.weights[j] + learning_rate * error * sigmoidDerivative(hiddenOutputs[k][i]) * hiddenOutputs[k - 1][j]
            end
            cell.bias = cell.bias + learning_rate * error * sigmoidDerivative(hiddenOutputs[k][i])
        end
    end
end

-- Shuffle the dataset
local function shuffleDataset(inputs, targets)
    local shuffledInputs = {}
    local shuffledTargets = {}

    for i = 1, #inputs do
        local randomIndex = math.random(1, #inputs)
        table.insert(shuffledInputs, inputs[randomIndex])
        table.insert(shuffledTargets, targets[randomIndex])
    end

    return shuffledInputs, shuffledTargets
end

-- Define a function to split dataset into mini-batches
local function splitIntoBatches(inputs, targets, batchSize)
    local batchesInputs = {}
    local batchesTargets = {}

    for i = 1, #inputs, batchSize do
        local batchInputs = {}
        local batchTargets = {}
        for j = i, math.min(i + batchSize - 1, #inputs) do
            table.insert(batchInputs, inputs[j])
            table.insert(batchTargets, targets[j])
        end
        table.insert(batchesInputs, batchInputs)
        table.insert(batchesTargets, batchTargets)
    end

    return batchesInputs, batchesTargets
end

-- Define function for training using SGD
function NeuralNetwork:trainSGD(inputs, targets, learningRate, epochs, batchSize)
    for epoch = 1, epochs do
        -- Shuffle the dataset
        local shuffledInputs, shuffledTargets = shuffleDataset(inputs, targets)

        -- Split dataset into mini-batches
        local batchesInputs, batchesTargets = splitIntoBatches(shuffledInputs, shuffledTargets, batchSize)

        -- Update weights for each mini-batch
        for i = 1, #batchesInputs do
            local batch_inputs = batchesInputs[i]
            local batch_targets = batchesTargets[i]
            for j = 1, #batch_inputs do
                self:updateWeights(batch_inputs[j], batch_targets[j], learningRate)
            end
        end
    end
end

function NeuralNetwork:calculateCost(inputs, targets)
    local totalCost = 0
    for i, input in ipairs(inputs) do
        local outputs = self:feedforward(input)
        for j = 1, #outputs do
            totalCost = totalCost + (targets[i][j] - outputs[j])^2
        end
    end
    return totalCost / #inputs
end

-- Example usage
math.randomseed(os.time())  -- Seed random number generator

local inputNodes = 2
local hiddenNodesList = {3, 4} -- Number of hidden nodes in each hidden layer
local outputNodes = 1

local nn = NeuralNetwork:new(inputNodes, hiddenNodesList, outputNodes)

-- Example training data
local inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
local targets = {{0}, {1}, {1}, {0}}

-- Example training loop
local epochs = 1000
local learningRate = 0.1
local batchSize = 100

nn:trainSGD(inputs, targets, learningRate, epochs, batchSize)

-- Calculate and print cost
local cost = nn:calculateCost(inputs, targets)
print("Mean squared error after training:", cost)
