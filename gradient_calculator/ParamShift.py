from hmac import new
from typing import Any, List, Tuple, Callable, Union

import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq

from gradient_calculator import CustomDifferentiator


class ParamShift:
    EXACT = "exact"
    SAMPLED = "sampled"

    def __init__(self, circuit: cirq.Circuit, observable: cirq.Pauli, F: Callable, expectation_mode="sampled"):
        self.circuit = circuit
        self.qubit_number = len(circuit.all_qubits())
        self.observable = observable
        self.F = F
        if expectation_mode == self.SAMPLED:
            self.expectation = self._get_noisy_sampler()
        elif expectation_mode == self.EXACT:
            self.expectation = self._get_exact_sampler()
        else:
            raise Exception(f"unknown expectation mode {expectation_mode}")

    def _get_noisy_sampler(self) -> tf.keras.layers.Layer:
        return tfq.layers.SampledExpectation(
            differentiator=tfq.differentiators.ParameterShift())

    def _get_exact_sampler(self) -> tf.keras.layers.Layer:
        return tfq.layers.Expectation(
            differentiator=CustomDifferentiator.CustomDifferentiator()
        )

    def get_expectation(self, symbol_names: List[Any], symbol_values: List[Any]) -> tf.Tensor:
        return self.expectation(self.circuit,
                                operators=self.observable,
                                symbol_names=symbol_names,
                                symbol_values=symbol_values)

    def get_gradient(self, symbol_names: List[Any], in_values: tf.Tensor, train_y: Any) -> tuple[
        Any, Any, Union[int, Any]]:
        mse = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as g:
            derivative_values = self.get_derivative_circs(symbol_names, in_values)
            g.watch(in_values)
            in_more_values = tf.concat([in_values, derivative_values], axis=0)
            out = self.expectation(
                self.circuit,
                operators=self.observable,
                symbol_names=symbol_names,
                symbol_values=in_more_values)
            # loss = mse(train_y, out)
            space_size = in_values.shape[0]

            f = out[:space_size]
            df_x = out[space_size:]
            df_x = np.pi / 2 * (df_x[::2] - df_x[1::2])
            df_x = tf.reduce_mean([df_x[i::self.qubit_number] for i in range(self.qubit_number)], axis=0)
            x = tf.reshape(train_y, (space_size, 1))

            loss = mse(self.F(df_x, f, x), 0.0) + (f[0,0] - 1) ** 2 # first entry should be 0

            grad = g.gradient(loss, in_more_values)
        # tf.transpose(grad)


            #loss = mse(f, 0)
            #print(loss)
            #print(loss.shape)
            #the_grad = g.gradient(loss, in_values)
            #print(the_grad.shape)
        return f, df_x, loss, grad

    def get_derivative_circs(self, symbol_names: List[Any], in_values: tf.Tensor) -> List[tf.Tensor]:
        more_values = []
        for i in range(self.qubit_number):
            ### Derivate the i-th qubit
            x_symbol_offset = len(symbol_names) - self.qubit_number + i
            values = in_values.numpy()
            mask = np.zeros(len(symbol_names))
            mask[x_symbol_offset] = 1

            forward_values = values + 0.5 * mask
            backward_values = values - 0.5 * mask

            more_values += [forward_values, backward_values]

        more_values = np.array(more_values)
        return tf.convert_to_tensor(
            more_values.reshape(more_values.shape[0] * more_values.shape[1], more_values.shape[2]), dtype=tf.float32)
