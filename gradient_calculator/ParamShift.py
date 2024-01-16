from hmac import new
from typing import Any, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq


class ParamShift:
    EXACT = "exact"
    SAMPLED = "sampled"

    def __init__(self, circuit: cirq.Circuit, observable: cirq.Pauli, expectation_mode="sampled"):
        self.circuit = circuit
        self.observable = observable
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
            differentiator=tfq.differentiators.ParameterShift())

    def get_expectation(self, symbol_names: List[Any], symbol_values: List[Any]) -> tf.Tensor:
        return self.expectation(self.circuit, 
                    operators=self.observable,
                    symbol_names=symbol_names, 
                    symbol_values=symbol_values)

    def get_gradient(self, symbol_names: List[Any], in_values: tf.Tensor, train_y: Any) -> tuple[Any, Any]:
        mse = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as g:
            g.watch(in_values)
            out = self.expectation(
                self.circuit,
                operators=self.observable,
                symbol_names=symbol_names,
                symbol_values=in_values)
            grad = g.gradient(out, in_values)
        #tf.transpose(grad)

        return out, grad * tf.reshape(out, (len(out), 1)) - 2 * grad * tf.reshape(train_y, (len(out), 1))
