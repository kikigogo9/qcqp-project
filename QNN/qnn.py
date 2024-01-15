import numpy as np
import tensorflow as tf
from tqdm import tqdm

from QNN.cost_function.cost_function import CostFunction
from QNN.cost_function.ising_hamiltonian import IsingHamiltonian

qubit_number = 5
depth = 5
epoch = 1000
space_size = 100

seed = 42


def build_binary_table(space, depth):
    """
    Depth defines the approximation of the binary number.
    Depth=5 means we take the 5 most significant digits of the input float
    :param space:
    :param depth:
    :return:
    """
    binary_table = tf.convert_to_tensor([])
    for x_in in space:
        x_1 = []
        for _ in range(depth):
            x_in = 2 * x_in
            x_1 += [np.floor(2 * x_in)]
            if x_in >= 1:
                x_in -= 1
        binary_table = tf.concat(binary_table, tf.convert_to_tensor(x_1))
    return binary_table


@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

if __name__ == "__main__":
    # get circuit of feature map
    # get circuit of ansatz
    cost_function: CostFunction = IsingHamiltonian()
    theta = tf.random.uniform(qubit_number * depth, 0, 2 * np.pi, seed=42)

    theta_hist = []
    loss_hist = []

    train_X = tf.linspace(0, 1, space_size)
    train_Y = tf.cos(np.pi * train_X)
    initial_value = 1.0

    x_bin_table = build_binary_table(train_X, depth)

    for i in tqdm(range(epoch)):
        Y_pred = tf.zeros()
        for x in range(train_X):
            x_bin = x_bin_table[x]
            Y_pred[x] = cost_function.get_cost(tf.concat(x_bin, theta), initial_value)

