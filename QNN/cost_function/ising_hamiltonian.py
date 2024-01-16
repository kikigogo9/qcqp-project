from random import Random
from typing import Any, List, Tuple

import cirq
import numpy as np
import sympy
from cirq import Circuit
import tensorflow as tf

from QNN.ansatz.ansatz import Ansatz
from QNN.cost_function.cost_function import CostFunction
from gradient_calculator.ParamShift import ParamShift


class IsingHamiltonian(CostFunction):
    """
    This class produces a more sophisticated version of the other cost functions.
    """

    def __init__(self, function: Circuit, qubits, symbol_names: List[Any]):
        """

        @param function: ODE circuit we currently train
        @param symbol_names: List of sympy variable names
        """
        super().__init__(function)
        self.symbol_names = symbol_names
        print("symbol_name_size" + str(len(symbol_names)))
        self.cost_function = None
        for i, q in enumerate(qubits):
            if self.cost_function is None:
                self.cost_function = self._random_weight() * cirq.Z(q)
            else:
                self.cost_function += cirq.Z(q)
            #self.cost_function += self._random_weight() * cirq.X(q)
            #for j, q2 in enumerate(qubits):
            #    if j < i:
            #        self.cost_function += cirq.Z(q2) * cirq.Z(q) * self._random_weight()*2

        self.expectation = ParamShift(function, self.cost_function, ParamShift.EXACT)

    def get_cost(self, in_values: Any) -> tf.Tensor:
        """

        @param in_values: parameters included in the circuit
        @return: cost tensor of the circuit with the given parameters
        """
        return self.expectation.get_expectation(
            self.symbol_names,
            in_values)
    
    def get_gradient_cost(self, in_values: Any, train_y: Any) -> tuple[Any, Any]:
        """
        
        @param in_values: parameters included in the circuit
        @return: cost tensors of the gradient of the circuit with the given parameters
        """
        return self.expectation.get_gradient(
            self.symbol_names,
            in_values,
            train_y
        )

    def _random_weight(self) -> float:
        return 0.5 * float(np.random.choice([-1, 1]) * np.random.normal(1, 0.33))


if __name__ == "__main__":
    a, b = sympy.symbols('a b')
    q0, q1 = cirq.GridQubit.rect(1, 2)

    #function = Circuit(
    #    cirq.rx(a).on(q0),
    #    cirq.rx(b).on(q1), cirq.CNOT(q0, q1)
    #)
    ansatz = Ansatz.generate_circuit(3, 2)

    cost_function = IsingHamiltonian(ansatz.circuit, ansatz.qubits, ansatz.symbol_names)
    in_values = np.random.random(len(ansatz.symbol_names))

    print(cost_function.get_cost([in_values]))