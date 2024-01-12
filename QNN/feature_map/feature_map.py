from abc import ABC, abstractmethod

import numpy as np
import sympy
import cirq
import tensorflow_quantum as tfq

class featureMap(ABC):
    @abstractmethod
    def __init__(self, circuit : cirq.Circuit, symbol : sympy.Symbol, qubits):
        self.circuit = circuit
        self.qubits = qubits
        self.nQubit = len(circuit.all_qubits())
        self.symbol = symbol

    @abstractmethod
    def rotationFunction(self, index : int):
        pass
    
    # Apply parametrized rotation gates for N qubits
    # tower denotes whether we are working with a regular product/chebyshev feature map
    # or a product/chebyshev tower feature map
    def parametrizedCircuit(self, tower : bool):
        # Apply a parametrized rotation gate for each qubit
        for i in range(self.nQubit):
            self.circuit.append(cirq.ry(self.rotationFunction(i if tower else 1 / 2)).on(self.qubits[i]))
        return tfq.convert_to_tensor([self.circuit])

class productMap(featureMap):
    # Nonlinear rotation function for product map
    def rotationFunction(self, index : int):
        return 2*index*np.arcsin(self.symbol)
    
class chebyshevMap(featureMap):
    # Nonlinear rotation function for chebyshev map
    def rotationFunction(self, index : int):
        return 2*index*np.arccos(self.symbol)
