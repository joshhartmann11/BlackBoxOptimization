"""
Gradient descent uses approximations of gradients to minimize a function.
"""

import black_box_optimization as bbo

class GradientDescent(bbo.Optimization):

    RELATIVE_EPSILON = 0.00001
    MOMENTUM_INCREMENT = lambda self, x: 0.1 / (x + 0.1)**2

    def optimize(self, m=1000, step=0.2, momentum=True):
        # Construct momentum structures for the
        prev_grad = {}
        momentum = {}
        for p in self.params:
            prev_grad[p] = 0
            momentum[p] = 1.0

        parameter_num = 0

        for i in range(int(m / 2)):
            p = self.params[parameter_num]
            mom = momentum[p]
            p_grad = prev_grad[p]
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
            if momentum:
                if grad * p_grad > 0: # both positive or both negative
                    mom += self.MOMENTUM_INCREMENT(mom)
                else:
                    mom = mom / 2

            # Update value
            p.value -= (grad * mom) * step - epsilon
            if (p.value > p.max):
                p.value = p.max
            elif (p.value < p.min):
                p.value = p.min

            momentum[p] = mom
            prev_grad[p] = grad
            parameter_num = (parameter_num + 1) % len(self.params)
        return self.params


def run(func, params, itterations=1000, step=0.2, momentum=True):
    optimizer = GradientDescent(func, params)
    optimizer.optimize(m=itterations, step=step, momentum=momentum)
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
