"""
This algorithm performs random normal perturbations on a random ever-decreasing
set of input parameters.  If the random perturbation performs better than the previous
parameters then the perturbation is kept.

This algorithm is written from
    Paper: "Dynamically dimensioned search algorithm for computationally efficient watershed model calibration"
    Date: January 2007
    Available: https://www.cs.ubc.ca/~hutter/EARG.shtml/earg/papers09/2005WR004723.pdf
    Accessed: August 19, 2019
    Authors: Bryan A. Tolson and Christine A. Shoemaker
"""

import random
import math

import black_box_optimization as bbo

class DynamicallyDimensionedSearch(bbo.Optimization):

    """
    args:
        m: number of itterations
        r: radius for variable perturbing
    """
    def optimize(self, m=1000, r=0.2):
        # Make an initial evalutation
        best_solution = self._run()
        self.error = best_solution
        prev_params = self._params_to_tuple()
        curr_params = list(prev_params)

        # For every itteration
        for i in range(m - 1):
            # Calculate the probability each parameter will be perturbed
            p_included = 1 - math.log(i + 1) / math.log(m)
            # Perturb parameters
            for j, p in enumerate(self.params):
                # At least one parameter has to be changed
                at_least_one = True
                while (at_least_one):
                    # Determine if it is to be perturbed
                    if random.random() < p_included:
                        at_least_one = False
                        # Calculate the change
                        value = curr_params[j]
                        value += random.gauss(0, 1) * r * (p.max - p.min)
                        # Reflect at variable boundaries
                        if value > p.max:
                            value = p.max - (value - p.max)
                            if value < p.min:
                                value = p.max
                        elif value < p.min:
                            value = p.min + (p.min - value)
                            if value > p.max:
                                value = p.min

                        curr_params[j] = value

            # Get the solution at the changed parameters
            solution = self._run(curr_params)
            if solution > best_solution: # Worse, restore parameters
                curr_params = list(prev_params)
            else: # Better, keep parameters
                best_solution = solution
                prev_params = tuple(curr_params)

        self.error = best_solution
        self._tuple_to_params(curr_params)

        return self.params


def run(func, params, itterations=1000, **kwargs):
    optimizer = DynamicallyDimensionedSearch(func, params)
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
