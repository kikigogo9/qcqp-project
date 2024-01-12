from abc import ABC, abstractmethod

from cirq import Circuit
import tensorflow as tf



class CostFunction(ABC):

    def __init__(self, function : Circuit):
        self.function = function
        self.nQubit = len(function.all_qubits())

    @abstractmethod
    def get_cost(self, in_values: tf.Tensor, initial_value: float):
        pass