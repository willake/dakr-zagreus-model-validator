local matrix = require("libs.lua-matrix.matrix")

-- Sigmoid function
local function sigmoid(z)
    return 1.0 / (1.0 + math.exp(-z))
end

-- Derivative of the sigmoid function
local function sigmoid_prime(z)
    return sigmoid(z) * (1 - sigmoid(z))
end

local function argmax(tensor)
    local max_val = -math.huge
    local max_index = -1
    for i = 1, tensor:size(1) do
        if tensor[i][1] > max_val then
            max_val = tensor[i][1]
            max_index = i
        end
    end
    return max_index
end

-- Box-Muller transform to generate random numbers from a normal distribution
local function randn(mu, sigma)
    local u1 = math.random()
    local u2 = math.random()
    local z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return z0 * sigma + mu
end

-- Shuffle the dataset with Fisher–Yates shuffle
local function shuffle(data)
    for i = #data, 2, -1 do
        local randomIndex = math.random(1, i)
        data[i], data[randomIndex] = data[randomIndex], data[i]
    end

    return data
end

-- Network class
Network = {}
Network.__index = Network

function Network.new(sizes)
    local self = setmetatable({}, Network)
    self.num_layers = #sizes
    self.sizes = sizes
    self.biases = {}
    self.weights = {}
    -- generate random biases
    for i = 2, #sizes do
        local layer_biases = {}
        for j = 1, sizes[i] do
            table.insert(layer_biases, math.random()) 
        end
        table.insert(self.biases, layer_biases)
    end
    -- generate random weights
    for i = 2, #sizes do
        local layer_weights = {}
        for j = 1, sizes[i] do
            local cell_weights = {}
            for k = 1, sizes[i - 1] do
                table.insert(cell_weights, math.random())
            end
            table.insert(layer_weights, cell_weights)
        end
        table.insert(self.weights, layer_weights)
    end
    return self
end

function Network.feedforward(self, a)
    local result = {}
    for i = 1, #self.biases do
        table.insert(result, sigmoid(matrix.mul(self.weights[i], a) + self.biases[i]))
    end
    return a
end

function Network.SGD(self, training_data, epochs, mini_batch_size, eta, test_data)
    local n = #training_data
    local n_test = 0

    if test_data then
        n_test = #test_data
    end
    
    for j = 1, epochs do
        shuffle(training_data)
        local mini_batches = {}
        for k = 1, n, mini_batch_size do
            local mini_batch = {}
            for l = k, math.min(k + mini_batch_size - 1, n) do
                table.insert(mini_batch, training_data[l])
            end
            table.insert(mini_batches, mini_batch)
        end

        for k = 1, #mini_batches do
            self:update_mini_batch(mini_batches[k], eta) 
        end
        
        if test_data then
            print("Epoch ", j, " : ", self:evaluate(test_data), " / ", n_test)
        else
            print("Epoch ", j, " complete")
        end
    end
end

function Network.update_mini_batch(self, mini_batch, eta)
    local nabla_b = matrix:new(#self.biases, #self.biases[1], 0)
    local nabla_w = {}
    for i = 1, #self.weights do
        table.insert(nabla_w, matrix:new(#self.weights[i], #self.weights[i][1], 0))
    end
    for i = 1, #mini_batch do
        local delta_nabla_b, delta_nabla_w = self:backprop(mini_batch[i][1], mini_batch[i][2])
        for j = 1, #nabla_b do
            nabla_b[j] = nabla_b[j] + delta_nabla_b[j]
            nabla_w[j] = nabla_w[j] + delta_nabla_w[j]
        end
    end
    for i = 1, #self.weights do
        self.weights[i] = self.weights[i] - (eta / #mini_batch) * nabla_w[i]
        self.biases[i] = self.biases[i] - (eta / #mini_batch) * nabla_b[i]
    end
end

function Network.backprop(self, x, y)
    local nabla_b = matrix:new(#self.biases, #self.biases[1], 0)
    local nabla_w = {}
    for i = 1, #self.weights do
        table.insert(nabla_w, matrix:new(#self.weights[i], #self.weights[i][1], 0))
    end
    -- Feedforward
    local activation = x
    local activations = {x}
    local zs = {}
    for i = 1, #self.biases do
        local z = matrix.mul(self.weights[i], activation) + self.biases[i]
        table.insert(zs, z)
        activation = sigmoid(z)
        table.insert(activations, activation)
    end
    -- Backward pass
    local delta = self:cost_derivative(activations[#activations], y):cmul(sigmoid_prime(zs[#zs]))
    nabla_b[#nabla_b] = delta
    nabla_w[#nabla_w] = matrix.mul(delta, activations[#activations-1]:transpose())
    for l = #self.biases - 1, 1, -1 do
        local z = zs[l]
        local sp = sigmoid_prime(z)
        delta = matrix.mul(matrix.transpose(self.weights[l+1]) * delta, sp)
        nabla_b[l] = delta
        nabla_w[l] = matrix.mul(delta, activations[l]:transpose())
    end
    return nabla_b, nabla_w
end

function Network.evaluate(self, test_data)
    local test_results = {}
    for i = 1, #test_data do
        table.insert(test_results, {argmax(self:feedforward(test_data[i][1])), test_data[i][2]})
    end
    local sum = 0
    for i = 1, #test_results do
        if test_results[i][1] == test_results[i][2] then
            sum = sum + 1
        end
    end
    return sum
end

function Network.cost_derivative(self, output_activations, y)
    return (output_activations - y)
end

-- Example usage:
-- Define your training data and test data
-- net = Network.new({input_size, hidden_size, output_size})
-- net:SGD(training_data, epochs, mini_batch_size, eta, test_data)