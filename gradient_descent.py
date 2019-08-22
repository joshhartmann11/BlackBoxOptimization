"""
Gradient descent uses approximations of gradients to minimize the function.
Implemented with Adam for faster, more robust, convergence
"""

import black_box_optimization as bbo
import math

class GradientDescent(bbo.Optimization):

    RELATIVE_EPSILON = 0.00001

    """
    args:
        m: Itterations
        alpha: learning rate
        beta: momentum rate
        gamma: RMS rate
    """
    def optimize(self, m=1000, alpha=0.3, beta=0.9, gamma=0.999):
        # Construct momentum structure
        momentum_coeffs = {}
        rms_coeffs = {}
        for p in self.params:
            momentum_coeffs[p] = 0
            rms_coeffs[p] = 0

        parameter_num = 0

        for i in range(int(m / 2)):
            p = self.params[parameter_num]
            mom = momentum_coeffs[p]
            rms = rms_coeffs[p]

            # Calculate the value
            t1 = self._run()
            self.error = t1

            # Add epsilon to parameters
            epsilon = (p.max - p.min) * self.RELATIVE_EPSILON
            if (p.value + epsilon > p.max):
                epsilon = -epsilon # Subtract instead
            p.value += epsilon

            # Calculate gradient
            t2 = self._run()
            grad = (t2 - t1) / epsilon

            # Update momentum
            rms = (gamma * rms) + ((1 - gamma) * grad**2)
            rms_coeffs[p] = rms

            # Update RMS
            mom = (beta * mom) + ((1 - beta) * grad)
            momentum_coeffs[p] = mom

            # Update value and constrain (Adam)
            p.value -= alpha * (mom) / (math.sqrt(rms) + epsilon) - epsilon
            if (p.value > p.max):
                p.value = p.max
            elif (p.value < p.min):
                p.value = p.min

            parameter_num = (parameter_num + 1) % len(self.params)

        return self.params


def run(func, params, itterations=1000, **kwargs):
    optimizer = GradientDescent(func, params)
    optimizer.optimize(m=itterations, **kwargs)
    return optimizer


if __name__ == "__main__":
    # Short test
    print("Testing...")
    function_to_minimize = lambda **kwargs: sum(kwargs.values())
    parameters = [
            { "name": "bob", "min": 0, "max": 100},
            { "name": "jim", "min": -100.5, "max": -50, "value": -70}
        ]
    print(run(function_to_minimize, parameters))
