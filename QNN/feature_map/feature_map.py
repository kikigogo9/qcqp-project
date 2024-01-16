from abc import ABC, abstractmethod

import numpy as np
import sympy
from sympy.stats import Arcsin
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

class featureMap(ABC):
    def __init__(self, symbol : sympy.Symbol, qubits):
        self.circuit = None
        self.qubits = qubits
        self.nQubit = len(qubits)
        self.symbol = symbol

class productMap(featureMap):
    """

    Implements product feature map with parametrized rotations
    """

    def rotationFunction(self, symbol):
        """

        Nonlinear rotation function for product map
        @return: rotation function
        """
        return symbol

    def parametrizedCircuit(self):
        """

        Apply parametrized rotation gates for N qubits
        @return: product map circuit
        """
        gates = []
        for i in range(self.nQubit):
            gates.append(cirq.ry((i+1)*self.rotationFunction(self.symbol)).on(self.qubits[i]))
        self.circuit = cirq.Circuit(*gates)
        return self.circuit
    
class chebyshevMap(featureMap):
    """

    Implements product chebyshev map
    """

    def rotationFunction(self, index : int):
        """

        Nonlinear rotation function for chebyshev map
        @param index: rotation (degree) index
        @return: rotation function
        """
        return np.cos(index * np.arccos(self.symbol))

    def parametrizedCircuit(self, degree: int = 2):
        """

        First apply hadamard to N qubits, then apply rotations with
        rotation function for a certain degree
        @param degree: depth of tower
        @return: chebyshev map circuit
        """
        for qubit in self.qubits:
            self.circuit.append(cirq.H(qubit))

        for i in range(degree):
            for j in range(self.nQubit):
                self.circuit.append(cirq.rx(self.rotationFunction(i) * np.pi).on(self.qubits[i]))
        return self.circuit

