import math
import random

def calculate_absolute_error(prediction, truth, output_count):
    sum_error = 0
    for i in range(output_count):  # only test first three outputs
        sum_error += abs(prediction[i] - truth[i])
    return sum_error / output_count

def calculate_squared_error(prediction, truth, output_count):
    sum_error = 0
    for i in range(output_count):  # only test first three outputs
        diff = truth[i] - prediction[i]
        sum_error += diff * diff
    return sum_error / output_count

def calculate_action_correctness(prediction, truth, action_count):
    p = 0
    pm = 0  # max of p
    t = 0
    tm = 0  # max of t
    for i in range(action_count):  # only test first three outputs
        if prediction[i] > pm:
            p = i
            pm = prediction[i]
        if truth[i] > tm:
            t = i
            tm = truth[i]
    return 1 if p == t else 0

def split_dataset_to_k_folds(dataset, k):
    n = len(dataset)
    part_size = n // k
    remainder = n % k
    start_index = 0
    splited = []
    for _ in range(k):
        size = part_size
        if remainder > 0:
            size += 1
            remainder -= 1
        part = dataset[start_index:start_index + size]
        splited.append(part)
        start_index += size
    return splited

def load_dataset_from_file(file_name):
    training_data = []
    with open(file_name, "r") as f:
        file_lines = f.readlines()

    for i in range(2, len(file_lines), 2):
        temp_inputs = [float(x) for x in file_lines[i].split()]
        temp_outputs = [float(x) for x in file_lines[i + 1].split()]
        training_data.append((temp_inputs, temp_outputs))
    return training_data