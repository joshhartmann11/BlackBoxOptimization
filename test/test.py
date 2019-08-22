import os
import sys
bbo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bbo_path)
import random
import statistics as stats

from black_box_optimization import Parameter
from gradient_descent import GradientDescent
from dynamically_dimensioned_search import DynamicallyDimensionedSearch
from combined_dds_gd import CombinedDDSGD

TEST_REPEAT = 1000
TEST_ITTERATIONS = 100
TEST_DIMENSIONS = 10


# Constant error, no change
def test_function_0(**kwargs):
    return 1;


# Returns sum of the absolute values of the parameters,
#   all parameters should decrease to their min
def test_function_1(**kwargs):
    return sum(kwargs.values())


# Convex, all parameters should converge to the middle of their range
def test_function_2(**kwargs):
    global test_class
    sum_conv = 0
    for p, v in zip(kwargs.keys(), kwargs.values()):
        variable = test_class.get_param_by_name(p)
        middle = (variable.max + variable.min) / 2
        slope = abs(variable.max / (abs(variable.min) + abs(variable.max))) + 0.1 # random gentle slope
        sum_conv += slope * (v - middle)**2 # Convex around middle
    return sum_conv


# Rosenbrock Function
#   f(x_0, x_1) = b(x_1 - x_0^2) + (a - x_0^2)^2
def test_function_3(**kwargs):
    global test_class
    fx = 0
    args = list(zip(kwargs.keys(), kwargs.values()))
    for i in range(int(len(args) / 2)):
        x0 = args[i * 2]
        p = test_class.get_param_by_name(x0[0])
        x0_n = x0[1] - ((p.max + p.min) / 2)
        if len(args) > (i + 1):
            x1 = args[i * 2 + 1]
            p = test_class.get_param_by_name(x1[0])
            x1_n = x1[1] - ((p.max + p.min) / 2)
        else:
            x1 = (0, "zero")
            x1_n = 0

        fx += 100 * (x1_n - x0_n**2) + (0 - x0_n)**2
    return fx


class TestProblem():

    TOLERANCE = 0.001

    def __init__(self, optimization, num_params):
        self.optimization = optimization
        self.params = []
        for i in range(num_params):
            name = str(i)
            scale = random.random() * 1000
            max = random.choice([-1, 1]) * random.random() * scale
            min = max - random.random() * scale
            value = random.random() * (max - min) + min
            self.params.append(Parameter(name, min, max, value))

    def score(self, param, value):
        return abs(param.value - value) / (param.max - param.min)

    def fuzzy_assert_equal(self, param, value):
        range = (param.max - param.min) * self.TOLERANCE
        assert (param.value > value - range)
        assert (param.value < value + range)

    def get_param_by_name(self, name):
        for sp in self.params:
            if (sp.name == name):
                return sp
        return None

    def test_0(self):
        optimizer = self.optimization(test_function_0, self.params)
        params = optimizer.optimize(TEST_ITTERATIONS)
        score = 0
        for p in params:
            sp = self.get_param_by_name(p.name)
            score += self.score(p, sp.value)
        return score / len(params)

    def test_1(self):
        optimizer = self.optimization(test_function_1, self.params)
        params = optimizer.optimize(TEST_ITTERATIONS)
        score = 0
        for p in params:
            score += self.score(p, p.min)
        return score / len(params)

    def test_2(self):
        optimizer = self.optimization(test_function_2, self.params)
        params = optimizer.optimize(TEST_ITTERATIONS)
        score = 0
        for p in params:
            score += self.score(p, (p.min + p.max) / 2)
        return score / len(params)

    def test_3(self):
        optimizer = self.optimization(test_function_3, self.params)
        params = optimizer.optimize(TEST_ITTERATIONS)
        score = 0
        for p in params:
            score += self.score(p, (p.min + p.max) / 2)
        return score / len(params)

    def run_test(self, test_num, num=1):
        tests = [self.test_0, self.test_1, self.test_2, self.test_3]
        try:
            print("test {}".format(test_num))
            scores = []
            for i in range(num):
                scores.append(tests[test_num]())
            print("  score avg: {:.6f}, stdev: {:.6f}".format(stats.mean(scores), stats.stdev(scores)))
            return stats.mean(scores)
        except AssertionError as e:
            print("  failed...")
            raise(e)

    def run_all(self, num=1):
        for i in range(4):
            self.run_test(i, num)



if __name__ == "__main__":
    print("Testing Gradient Descent...")
    test_class = TestProblem(GradientDescent, TEST_DIMENSIONS)
    test_class.run_all(TEST_REPEAT)

    print("\nTesting Dynamically Dimensioned Search...")
    test_class = TestProblem(DynamicallyDimensionedSearch, TEST_DIMENSIONS)
    test_class.run_all(TEST_REPEAT)
