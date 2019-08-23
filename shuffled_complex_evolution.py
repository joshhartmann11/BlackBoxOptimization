"""
Shuffled Complex Evolution: Just read the paper

This algorithm is based off of:
    Paper: "Shuffled complex evolution approach for effective and efficient global minimization"
    Date: January 1993
    Available: https://sci-hub.tw/https://link.springer.com/article/10.1007/BF00939380 (whoops)
    Accessed: August 19, 2019
    Authors: Duan, Q. Y., Gupta, V. K., & Sorooshian, S
"""

import copy
import random
import math

import black_box_optimization as bbo

class ShuffledComplexEvolution(bbo.Optimization):

    D = None

    def _weighted_sample(self, n, samples, weights):
        ret_samples = []
        ret_loc = []
        for i in range(n):
            choice = random.choices(k=1, population=range(len(weights)), weights=weights)[0] # choose something
            weights[choice] = 0 # remove the weight
            ret_samples.append(samples[choice])
            ret_loc.append(choice)
        return ret_samples, ret_loc


    """
    Yes this would be so much simpler with something like numpy or a vector representation
    Args:
        itt: minimum number of function calls
        p: number of complexes
        m: number of points in each complex
        q: number randomly selected from a complex
        alpha: number of times to mutate and kill offsprint
        beta: number of offspring to be generated per complex

    SCE1: p = 1
    SCE2: m = beta = 2n+1, alpha=1, where n = dimensions
    """
    def optimize(self, itt=1000, p=5, m=5, q=10, alpha=2, beta=2):
        func_calls = 0

        # Step 1: Generate m x p points and evaluate
        if not self.D:
            D = []
            for i in range(m * p):
                point = [0] * len(self.params)
                for j in range(len(self.params)):
                    param = self.params[j]
                    point[j] = (param.max - param.min) * random.random() + param.min
                D.append({"params": tuple(point), "result": self._run(point)})
                func_calls += 1
        else:
            D = self.D

        while (func_calls < itt):

            # Step 2: Rank points
            D.sort(key=lambda x : x["result"])

            # Step 3: Partition into p complexes
            complexes = []
            for k in range(p):
                complexes.append([D[k + p * j] for j in range(m) if (k + p * j) < len(D)])

            # Step 4: Evolve Each Complex CCE
            # Step 4.1: Assign weights
            weights = [2 * (m + 1 - i) / (m * (m + 1)) for i in range(1, m + 1)]
            for __askdf in range(beta):

                for A in complexes:
                    # Step 4.2: Choose q points based on probability distribution
                    B, L = self._weighted_sample(q, A, weights)

                    # Step 4.3: Generate Offspring

                    # 4.3 a) Compute Centroid in each dimension
                    centroid = [0] * len(self.params)

                    for point in B[:-1]:
                        # Vector addition
                        centroid = [c + p for c, p in zip(centroid, point["params"])]

                    # Vector multiplication
                    centroid = tuple([c * (1 / (len(B) - 1)) for c in centroid])

                    # 4.3 b) Calculate the new point
                    for __qjwhbf in range(alpha):
                        # Vector scaling and subtraction
                        r = tuple([2 * g - ug for g, ug in zip(centroid, B[-1]["params"])])

                        # 4.3 c) If r is in the problem space, chill, else mutate
                        if self._valid_point(r):
                            point_r = {"params": r, "result": self._run(r)}
                            func_calls += 1

                        else:
                            # Compute the smallest hypercube that contains all of B
                            mins = [float('inf')] * len(self.params)
                            maxs = [-float('inf')] * len(self.params)
                            for point in B:
                                params = point["params"]
                                for i in range(len(self.params)):
                                    if params[i] > maxs[i]:
                                        maxs[i] = params[i]
                                    elif params[i] < mins[i]:
                                        mins[i] = params[i]

                            # Generate a random value within the hypercube
                            point = [0] * len(self.params)
                            for i in range(len(self.params)):
                                point[i] = (maxs[i] - mins[i]) * random.random() + mins[i]
                            r = tuple(point)
                            point_r = {"params": r, "result": self._run(r)}
                            func_calls += 1

                        # 4.3 d) If the new point r is better than the last one in B,
                        #    set the point_B[-1] to r
                        if point_r["result"] < B[-1]["result"]:
                            B[-1] = point_r

                        # Else, compute point_c and evaluate
                        else:
                            c = tuple([(c + a) / 2 for c, a in zip(centroid, B[-1]["params"])])
                            point_c = {"params": c, "result": self._run(c)}
                            func_calls += 1

                            # 4.3 e) If the new point c is better than the last one in B,
                            #    set the point point_B[-1] to c
                            if point_c["result"] < B[-1]["result"]:
                                B[-1] = point_c
                            else:
                                mins = [float('inf')] * len(self.params)
                                maxs = [-float('inf')] * len(self.params)
                                for point in B:
                                    params = point["params"]
                                    for i in range(len(self.params)):
                                        if params[i] > maxs[i]:
                                            maxs[i] = params[i]
                                        elif params[i] < mins[i]:
                                            mins[i] = params[i]

                                # Generate a random value within the hypercube
                                point = [0] * len(self.params)
                                for i in range(len(self.params)):
                                    point[i] = (maxs[i] - mins[i]) * random.random() + mins[i]
                                z = tuple(point)
                                point_z = point_z = {"params": z, "result": self._run(z)}
                                func_calls += 1
                                B[-1] = point_z

                # Step 4.4: Replace Parents by offspring and sort A
                for i, l in enumerate(L):
                    A[l] = B[i]

                A.sort(key=lambda x: x["result"])

                # Step 4.5: Itterate

            # Step 5: Shuffle Complexes
            D = []
            for A in complexes:
                D.extend(A)

        self._tuple_to_param(D[0]["params"])
        self.error = D[0]["result"]
        self.D = D

        return self.params


if __name__ == "__main__":
    # Short test
    print("Testing...")
    function_to_minimize = lambda **kwargs: sum(kwargs.values())
    parameters = [
            { "name": "bob", "min": 0, "max": 100},
            { "name": "jim", "min": -100.5, "max": -50, "value": -70}
        ]
    optimizer = ShuffledComplexEvolution(function_to_minimize, parameters)
    optimizer.optimize()
    print(optimizer)
