import cirq


class Ansatz:
    def __init__(self, qubits: list[cirq.LineQubit], v_params: list):
        self.v_params = v_params  # Expected in radians
        self.depth = int(len(v_params) / 3)
        self.qubits = qubits
        self.nQubits = len(qubits)

    def hardware_efficient_rot(self):
        """
        The rotation layer of the hardware efficient ansatz. Apply Rz-Rx-Rz for independent
        angles given by v_params to all qubits.

        :return: cirq.Moment
        """
        rot_moment = cirq.Moment()
        for i in range(self.depth):
            r1 = cirq.rz(self.v_params[i])
            r2 = cirq.rx(self.v_params[i + 1])
            r3 = cirq.rz(self.v_params[i + 2])
            for j in range(self.nQubits):
                rot_moment.with_operation(r1.on(self.qubits[j]))
                rot_moment.with_operation(r2.on(self.qubits[j]))
                rot_moment.with_operation(r3.on(self.qubits[j]))
        return rot_moment

    def hardware_efficient_ent(self):
        """
        The entangling layer of the hardware efficient ansatz. Apply CNOT gates to all nearest
        neighbors.

        :return: cirq.Moment
        """
        ent_moment = cirq.Moment()
        for i in range(self.nQubits - 1):
            cirq.CNOT(self.qubits[i], self.qubits[i + 1])
        return ent_moment

    def hardware_efficient(self):
        """
        The full implementation of the hardware efficient ansatz. First apply the rotation layer
        to the qubits, followed by the entangling layer.

        :return: cirq.Circuit
        """
        return cirq.Circuit((self.hardware_efficient_rot(), self.hardware_efficient_ent()))
