import cirq
import string
import random
import numpy as np
import sympy
from cirq.contrib.svg import SVGCircuit


class Ansatz:
    def __init__(self, qubits: cirq.GridQubit, depth: int):
        self.depth = depth
        self.qubits = qubits
        self.nQubits = len(qubits)
        self.symbol_names = []
        self.circuit = None

    def hardware_efficient_rot(self):
        """
        The rotation layer of the hardware efficient ansatz. Apply Rz-Rx-Rz for independent
        angles given by v_params to all qubits.

        :return: cirq.Moment
        """
        rot_gates = []
        for i in range(self.depth):
            for j in range(self.nQubits):
                names = self.generate_names()
                self.symbol_names += names
                r1 = cirq.rz(names[0])
                r2 = cirq.rx(names[1])
                r3 = cirq.rz(names[2])

                rot_gates.append(r1.on(self.qubits[j]))
                rot_gates.append(r2.on(self.qubits[j]))
                rot_gates.append(r3.on(self.qubits[j]))
            for j in range(self.nQubits - 1):
                if j % 2 == 0:
                    rot_gates.append(cirq.CNOT(self.qubits[j], self.qubits[j + 1]))
            for j in range(self.nQubits - 1):
                if j % 2 == 1:
                    rot_gates.append(cirq.CNOT(self.qubits[j], self.qubits[j + 1]))
        return rot_gates

    def hardware_efficient(self):
        """
        The full implementation of the hardware efficient ansatz. First apply the rotation layer
        to the qubits, followed by the entangling layer.

        :return: cirq.Circuit
        """
        if self.circuit is None:
            self.circuit = cirq.Circuit(*self.hardware_efficient_rot())
        return self.circuit

    def generate_names(self) -> list[sympy.Symbol]:
        def generate_string(n: int) -> str:
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

        return sympy.symbols([generate_string(8) for _ in range(3)])

    def generate_circuit(qubits, qubit_number: int, depth: int):

        ansatz = Ansatz(qubits, depth)
        ansatz.hardware_efficient()
        return ansatz


if __name__ == "__main__":
    qubits = cirq.GridQubit.rect(5, 1)
    ansatz = Ansatz(qubits, 2)
    # print(SVGCircuit(ansatz.hardware_efficient())._repr_svg_())

    f = open("demofile2.svg", "w")
    f.write(SVGCircuit(ansatz.hardware_efficient())._repr_svg_())
    f.close()
