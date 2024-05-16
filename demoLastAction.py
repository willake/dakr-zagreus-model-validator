import torch
import torch.nn as nn
import random
import helper
import time

random.seed(89890)

learning_rate = 10  # set between 1, 100
epoch = 5  # number of times to do backpropagation
threshold = 1  # steepness of the sigmoid curve

k = 5
dataset = helper.load_dataset_from_file("DZrecord.log")

random.shuffle(dataset)

folds = helper.split_dataset_to_k_folds(dataset, k)  # for k-fold cross validation

# Define neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 6)
        self.fc4 = nn.Linear(6, 4)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Define the loss function
criterion = nn.MSELoss()

# run backpropagation (bp)
global_error_sum = 0  # MSE of actions probability
global_action_correctness_sum = 0  # Whether the predicted action holds highest probability is the same as ground truth
global_charge_time_error_sum = 0  # MSE of charge time
for test_idx in range(0, k):  # do k times
    network = NeuralNetwork()

    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

    start = time.time()
    for _ in range(epoch):
        for i in range(0, k):  # run through all folds
            if test_idx != i:  # if the index is not the test set, train model
                for data in folds[i]:
                    inputs = torch.tensor([data[0]], dtype=torch.float32)
                    labels = torch.tensor([data[1]], dtype=torch.float32)
                    optimizer.zero_grad()
                    outputs = network(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

    time_taken = time.time() - start

    # validate model using i-th fold, which is the training set
    test_set = folds[test_idx - 1]
    local_error_sum = 0
    local_action_correctness_sum = 0
    local_charge_time_error_sum = 0

    for data in test_set:
        inputs = torch.tensor([data[0]], dtype=torch.float32)
        labels = torch.tensor([data[1]], dtype=torch.float32)
        prediction = network(inputs).detach().numpy()[0]
        local_error_sum += helper.calculate_squared_error(prediction, data[1], 3)
        local_action_correctness_sum += helper.calculate_action_correctness(prediction, data[1], 3)
        local_charge_time_error_sum += helper.calculate_squared_error([prediction[3]], [data[1][3]], 1)  # 4th element is charge time

    local_error = local_error_sum / len(test_set)
    local_action_correctness = local_action_correctness_sum / len(test_set)
    local_charge_time_error = local_charge_time_error_sum / len(test_set)

    print(f"{test_idx}th iteration - training time: {time_taken:.2f}, action correctness: {local_action_correctness:.2f} ({local_action_correctness_sum}/{len(test_set)}), charge time error: {local_charge_time_error:.3f}")

    global_error_sum += local_error
    global_action_correctness_sum += local_action_correctness
    global_charge_time_error_sum += local_charge_time_error

error = global_error_sum / k
action_correctness = global_action_correctness_sum / k
charge_time_error = global_charge_time_error_sum / k

print(f"Cross-validation result - action MSE: {error:.3f}, action correctness: {action_correctness:.2f}, charge time MSE: {charge_time_error:.3f}")
