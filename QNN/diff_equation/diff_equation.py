from typing import Any
import tensorflow as tf

from QNN.cost_function.cost_function import CostFunction

class DiffEquation():
    """

    This class contains the differential equation, which can be solved using 
    the cost function, initial values and problem parameters initialized in __init__
    """
    def __init__(self, **kwargs):
        for variable, value in kwargs.iteritems():
            setattr(self, variable, value)
    
    def ODE(cost_function : CostFunction, in_values: Any) -> tf.Tensor:
        """
        
        @param cost_function: cost function that will be evaluated for different
        function/derivative expectation values and in_values
        @param in_values: parameters included in the circuits
        @return: solution tensor of the ODE
        """
        return cost_function.get_gradient_cost(tf.convert_to_tensor(in_values))
        + tf.math.scalar_mul(self.lamb, 
            cost_function.get_cost(tf.math.add(self.kappa, 
                tf.math.tan(tf.math.scalar_mul(self.lamb, in_values)))
            )
        )

    def ODE_sol(in_values: Any, initial_value: float) -> tf.Tensor:
        """

        @param in_values: parameters included in the circuits
        @param initial_value: initial value of the ODE
        @return: analytical solution tensor of the ODE
        """
        return tf.add(tf.math.multiply(tf.math.exp(tf.math.scalar_mul(-1*self.lamb*self.kappa, in_values)),
                                tf.math.cos(tf.math.scalar_mul(self.lamb, in_values))), initial_value)