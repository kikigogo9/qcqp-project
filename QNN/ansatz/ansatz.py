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
        print(self.nQubits)
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
            names = self.generate_names()
            self.symbol_names += names
            for j in range(self.nQubits):
                r1 = cirq.rz(names[j])
                r2 = cirq.rx(names[j])
                r3 = cirq.rz(names[j])

                rot_gates.append(r1.on(self.qubits[j]))
                rot_gates.append(r2.on(self.qubits[j]))
                rot_gates.append(r3.on(self.qubits[j]))
            for j in range(self.nQubits-1):
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

    def generate_names(self)->list[sympy.Symbol]:
        def generate_string(n:int)->str:
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

        return sympy.symbols([generate_string(8) for _ in range(self.nQubits)])

    def generate_circuit(qubit_number: int, depth:int):
        qubits = cirq.GridQubit.rect(qubit_number, 1)
        ansatz = Ansatz(qubits, depth)
        return ansatz


if __name__ == "__main__":
    qubits = cirq.GridQubit.rect(5,1)
    ansatz = Ansatz(qubits, 2)
    #print(SVGCircuit(ansatz.hardware_efficient())._repr_svg_())

    f = open("demofile2.svg", "w")
    f.write(SVGCircuit(ansatz.hardware_efficient())._repr_svg_())
    f.close()
