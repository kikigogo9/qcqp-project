from random import Random
from typing import Any

import cirq
import numpy as np
import sympy
from cirq import Circuit

from QNN.cost_function.cost_function import CostFunction
from gradient_calculator.ParamShift import ParamShift


class IsingHamiltonian(CostFunction):
    """
    This class produces a more sophisticated version of the other cost functions.
    """

    def __init__(self, function: Circuit, qubits, symbol_names, initial_value=0):
        """

        @param function: ODE circuit we currently train
        @param symbol_names: List of sympy variable names
        @param initial_value: boundary condition at f(x_0)
        """
        super().__init__(function)
        self.symbol_names = symbol_names
        self.initial_value = initial_value

        self.cost_function = None
        for q in qubits:
            if self.cost_function is None:
                self.cost_function = self._random_weight() * cirq.Z(q)
            else:
                self.cost_function += self._random_weight() * cirq.Z(q)
            self.cost_function += self._random_weight() * cirq.X(q)
            for q2 in qubits:
                if not (q2 is q):
                    self.cost_function += cirq.Z(q2) * cirq.Z(q) * self._random_weight()

        self.expectation = ParamShift(function, self.cost_function, ParamShift.EXACT)

    def get_cost(self, in_values: Any, initial_value: float = 0) -> float:
        """

        @param initial_value:
        @param in_values: parameters included in the circuit
        @return: cost value of the circuit with the given parameters
        """
        self.initial_value = initial_value
        return (initial_value + self.expectation.get_expectation(
            self.function,
            self.cost_function,
            self.symbol_names,
            in_values,
            None)
                )

    def _random_weight(self) -> float:
        return 2 * (Random().random() - 0.5)


if __name__ == "__main__":
    a, b = sympy.symbols('a b')
    q0, q1 = cirq.GridQubit.rect(1, 2)

    function = Circuit(
        cirq.rx(a).on(q0),
        cirq.rx(b).on(q1), cirq.CNOT(q0, q1)
    )
    cost_function = IsingHamiltonian(function, [q0, q1], [a, b])
    in_values = np.array([[np.pi, np.pi / 2.]])

    print(cost_function.get_cost(in_values))
