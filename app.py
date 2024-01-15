import cirq
from cirq.contrib.svg import SVGCircuit
from flask import Flask

from QNN.ansatz.ansatz import Ansatz

app = Flask(__name__)

@app.route('/home')
def hello_world():
    qubits = cirq.GridQubit.rect(5,1)
    ansatz = Ansatz(qubits, 5)
    print(SVGCircuit(ansatz.hardware_efficient())._repr_svg_())
    return SVGCircuit(ansatz.hardware_efficient())._repr_svg_()

@app.route('/')
def hello_world():
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)