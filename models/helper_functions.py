import numpy as np


def generate_input_dims(input_dim):
    input_dims_up = []
    initial_input_dim = np.array(input_dim)
    input_dims_down = []

    for i in range(4):
        input_dims_down.append((initial_input_dim // 2 ** i).tolist())
    reversed = input_dims_down[::-1]
    for idx, in_dim in enumerate(reversed[:-1]):
        input_dims_up.append([in_dim, reversed[idx + 1]])

    return input_dims_down + input_dims_up


def generate_multiple_input_dims(input_dims):
    return [generate_input_dims(in_dim) for in_dim in input_dims]
