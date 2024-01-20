import tensorflow_quantum as tfq
import tensorflow as tf
import numpy as np
from tensorflow_quantum.python.differentiators import parameter_shift_util


class CustomDifferentiator(tfq.differentiators.Differentiator):
    """
    A custom differentiator example
    """
    @tf.function
    def get_gradient_circuits(self, programs, symbol_names, symbol_values):
        """See base class description."""
        # these get used a lot
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)

        # Assume cirq.decompose() generates gates with at most two distinct
        # eigenvalues, which results in two parameter shifts.
        n_shifts = 2

        # These new_programs are parameter shifted.
        # shapes: [n_symbols, n_programs, n_param_gates, n_shifts]
        (new_programs, weights, shifts,
         n_param_gates) = parameter_shift_util.parse_programs(
             programs, symbol_names, symbol_values, n_symbols)

        m_tile = n_shifts * n_param_gates * n_symbols

        # Transpose to correct shape,
        # [n_programs, n_symbols, n_param_gates, n_shifts],
        # then reshape to the correct batch size
        batch_programs = tf.reshape(tf.transpose(new_programs, [1, 0, 2, 3]),
                                    [n_programs, m_tile])
        batch_weights = tf.reshape(
            tf.transpose(weights, [1, 0, 2, 3]),
            [n_programs, n_symbols, n_param_gates * n_shifts])
        shifts = tf.reshape(tf.transpose(shifts, [1, 0, 2, 3]),
                            [n_programs, m_tile, 1])

        # Append impurity symbol into symbol name
        new_symbol_names = tf.concat([
            symbol_names,
            tf.constant([parameter_shift_util.PARAMETER_IMPURITY_NAME])
        ], 0)

        # Symbol values are the input symbol values, tiled according to
        # `batch_programs`, with the shift values appended.
        tiled_symbol_values = tf.tile(tf.expand_dims(symbol_values, 1),
                                      [1, m_tile, 1])
        batch_symbol_values = tf.concat([tiled_symbol_values, shifts], 2)

        single_program_mapper = tf.reshape(
            tf.range(n_symbols * n_param_gates * n_shifts),
            [n_symbols, n_param_gates * n_shifts])
        batch_mapper = tf.tile(tf.expand_dims(single_program_mapper, 0),
                               [n_programs, 1, 1])

        return (batch_programs, new_symbol_names, batch_symbol_values,
                batch_weights, batch_mapper)