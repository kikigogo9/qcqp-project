from typing import Any, List
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq


class ParamShift:
    EXACT = "exact"
    SAMPLED = "sampled"

    def __init__(self, circuit: cirq.Circuit, observable: cirq.Pauli, expectation_mode="sampled"):

        self.circuit = circuit
        self.observable = observable
        if expectation_mode == "sampled":
            self.expectation = self._get_noisy_sampler()
        elif expectation_mode == "exact":
            self.expectation = self._get_exact_sampler()
        else:
            raise Exception(f"unknown expectation mode {expectation_mode}")

    def _get_noisy_sampler(self) -> tf.keras.layers.Layer:
        return tfq.layers.SampledExpectation(
            differentiator=tfq.differentiators.ParameterShift())

    def _get_exact_sampler(self) -> tf.keras.layers.Layer:
        return tfq.layers.Expectation(
            differentiator=tfq.differentiators.ParameterShift())

    def get_expectation(self,
                       circuit: cirq.Circuit,
                       operator: cirq.Pauli,
                       symbol_names: List[Any],
                       symbol_values: List[Any],
                       repetitions=500
                       ) -> float:
        return self.expectation(circuit, operators=operator, symbol_names=symbol_names, symbol_values=symbol_values)

    def get_gradient(self, in_values: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as g:
            g.watch(in_values)
            out = self.expectation(
                self.circuit,
                operators=self.observable,
                repetitions=500,
                symbol_names=['alpha'],
                symbol_values=in_values)

        return g.gradient(out, in_values)
