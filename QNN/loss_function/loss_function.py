from typing import Any

import cirq
import numpy as np
import sympy
from cirq import Circuit
import tensorflow as tf

from QNN.cost_function.cost_function import CostFunction
from QNN.diff_equation.diff_equation import DiffEquation

class lossFunction:
    """

    This class contains the loss function, which can be selected from initialization.
    To calculate the loss function, the cost function is given with a specific differential
    equation
    """
    MSE = "MSE" # Mean squared error
    MAE = "MAE" # Mean absolute error
    KLD = "KLD" # Kullback-Leibler (KL) divergence

    def __init__(self, function : CostFunction, loss="MSE"):
        self.cost_function = function
        if loss == self.MSE:
            self.loss = tf.keras.losses.MeanSquaredError()
        elif loss == self.MAE:
            self.loss = tf.keras.losses.MeanAbsoluteError()
        elif loss == self.KLD:
            self.loss = tf.keras.losses.KLDivergence()
        else:
            raise Exception(f"unknown loss function {loss}")

    def get_loss(self, in_values: Any, equation: DiffEquation):
        """
        
        @param in_values: parameters included in the circuits
        @param equation: differential equation class that will compute
        the solutions using the cost function and in_values
        @return: loss of particular differential equation & cost function
        with given input values, using specified loss function
        """
        solutions = equation.ODE(self.cost_function)
        return self.loss(solutions, tf.zeros(tf.shape(solutions)))