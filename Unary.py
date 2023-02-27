import numpy as np
import random
import math
import numbers


class Unary():
    def __init__(self, e, d, optimized=True, p=0, q=0):
        self.d = d
        self.p = p
        self.q = q
        self.e = e
        self.optimized = optimized
        if self.optimized:
            self.p = 1 / 2
            self.q = 1 / (math.exp(self.e) + 1)

    def encode(self, v):
        assert (v < self.d)
        B = np.zeros(self.d)
        B[v] = 1
        return B

    def perturb(self, ret):
        B = ret

        new_B = B
        for i in range(len(B)):
            if B[i] == 1:
                pr = self.p
            else:
                pr = self.q
            res = random.random()
            if res < pr:
                new_B[i] = 1
            else:
                new_B[i] = 0

        return new_B

    def randomize(self, v):
        return self.perturb(self.encode(v))

    def aggregate(self, config):
        reported_values = config['reported_values']
        d = config['d']

        p = self.p
        q = self.q

        results = np.zeros(d)
        n = len(reported_values)

        for i in range(d):
            sum_v = 0
            for j in reported_values:
                if j[i] == 1:
                    sum_v += 1
            results[i] = ((sum_v) - n * q) / (p - q)
            if (results[i] < 0):
                results[i] = 0

        return results