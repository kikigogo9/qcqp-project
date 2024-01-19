from copy import deepcopy

import cirq
import numpy as np
import sympy
import tensorflow as tf
from tqdm import tqdm

from QNN.ansatz.ansatz import Ansatz
from QNN.cost_function.cost_function import CostFunction
from QNN.cost_function.ising_hamiltonian import IsingHamiltonian
from QNN.feature_map.feature_map import productMap
import matplotlib.pyplot as plt

qubit_number = 4
depth = 4
epoch = 100
space_size = 10
learning_rate = 0.05
seed = 42


def plot_fit_and_loss():
    plt.plot(train_X, Y_pred)
    plt.plot(train_X, train_Y)
    plt.plot(train_X, best_fit, linestyle='dashed')
    plt.savefig("fit.png")
    plt.plot(range(epoch), loss_hist)
    plt.yscale("log")
    plt.plot(tf.transpose(theta_hist)[0], linestyle='dashed')
    plt.plot(tf.transpose(theta_hist)[2], linestyle='dashed')
    plt.plot(tf.transpose(theta_hist)[1], linestyle='dashed')
    plt.savefig("loss.png")


if True:

    qubits = cirq.GridQubit.rect(qubit_number, 1)
    x = sympy.symbols(["x" + str(i) for i in range(qubit_number)])
    feature_map = productMap(symbols=x, qubits=qubits).parametrizedCircuit()
    ansatz = Ansatz.generate_circuit(qubits, qubit_number, depth)

    cost_function = IsingHamiltonian(feature_map + ansatz.circuit, ansatz.qubits, ansatz.symbol_names + x)
    in_values = np.random.random(len(ansatz.symbol_names))

    theta = tf.convert_to_tensor(np.random.uniform(0, 2 * np.pi, (qubit_number * depth * 3)), dtype=tf.float32)

    theta_hist = []
    loss_hist = []
    best_fit = None
    best_loss = 100

    train_X = tf.convert_to_tensor(np.linspace(0, 1, space_size, dtype=np.float32))
    train_Y = tf.sin(2* np.pi*train_X)

    loss = tf.keras.losses.MeanSquaredError()
    for i in tqdm(range(epoch)):


        grads = []

        #new_values = tf.convert_to_tensor([tf.concat([theta, train_X], axis=0)])
        new_values = tf.repeat([theta], repeats=space_size, axis=0)

        new_values = tf.transpose(new_values)
        new_values = tf.concat([new_values, tf.reshape(tf.repeat([train_X], repeats=qubit_number, axis=0), [qubit_number, space_size])], axis=0)
        new_values = tf.transpose(new_values)
        
        

        Y_pred, gradient = cost_function.get_gradient_cost(new_values, train_Y)
        grads.append(gradient)
        reduced = tf.reduce_mean(tf.convert_to_tensor(grads), axis=1)
        theta -= learning_rate * reduced[0, :-qubit_number]

        plt.savefig("asd.png")
        theta_hist += [deepcopy(theta)]
        l = loss(train_Y, Y_pred)
        if best_loss > l:
            best_loss = l
            best_fit = Y_pred
        loss_hist += [l]

    plot_fit_and_loss()
    
