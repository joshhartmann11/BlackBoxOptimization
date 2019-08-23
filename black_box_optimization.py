

class Parameter():

    def __init__(self, name=None, min=None, max=None, value=None):
        self.name = name
        self.min = min
        self.max = max
        if min and max:
            if value:
                assert(value <= max)
                assert(value >= min)
                self.value = value
            else:
                self.value = (max - min) / 2 + min

    def from_dict(self, dict):
        self.name = dict["name"]
        self.min = dict["min"]
        self.max = dict["max"]
        if "value" in dict.keys():
            self.value = dict["value"]
        else:
            self.value = (self.min + self.max) / 2

    def to_dict(self):
        j = {}
        j["name"] = self.name
        j["min"] = self.min
        j["max"] = self.max
        j["value"] = self.value
        return j

    def __str__(self):
        return str(self.to_dict())


class Optimization():

    def __init__(self, func, params):
        self.func = func
        self.params = []
        for p in params:
            if type(p) is dict: # Convert from dict
                po = Parameter()
                po.from_dict(p)
                self.params.append(po)
            else:
                self.params.append(p)
        self.params.sort(key=lambda x: x.name)
        self.error = float('inf')

    def __str__(self):
        str = "Minimum: {:.5f}\nValues:\n".format(self.error)
        str += "\n".join(["  {}: {}".format(p.name, p.value) for p in self.params])
        return str

    def _params_to_tuple(self):
        pt = []
        for p in self.params:
            pt.append(p.value)
        pt = tuple(pt)
        return pt

    def _tuple_to_params(self, t):
        for i in range(len(self.params)):
            self.params[i].value = t[i]


    def _valid_point(self, point):
        for i in range(len(self.params)):
            if point[i] < self.params[i].min or \
               point[i] > self.params[i].max:
                return False
        return True

    def _run(self, params=None):
        kwargs = {}
        if not params:
            for p in self.params:
                kwargs[p.name] = p.value
        else:
            for i in range(len(self.params)):
                kwargs[self.params[i].name] = params[i]
        return self.func(**kwargs)

    def optimize(self):
        raise ImplementationError("\"optimize\" needs to be implemented")


if __name__ == "__main__":
    print("Testing...")
    p1 = Parameter("one", 0, 1, 0.5)
    assert(p1.value == 0.5)
    p2 = Parameter("two", -1, 1)
    assert(p2.value == 0)
    p3 = Parameter("three", -10, 1)
    assert(p3.value == -4.5)
    o = Optimization(lambda **kwargs: 1, [p1, p2, p3])
    assert(1 == o._run())
