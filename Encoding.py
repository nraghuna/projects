import numpy as np
import random
import math


class Encoding():
    def __init__(self, e, d):
        self.e = e
        self.d = d
        self.p = math.exp(self.e) / (math.exp(self.e) + self.d - 1)
        self.q = 1 / (math.exp(self.e) + self.d - 1)

    def encode(self, v):
        return v

    def perturbe(self, ret):
        x = ret
        res = random.random()
        if (res < self.p):
            pert = x
        else:
            false_xs = [i for i in range(self.d) if i != x]

            pert = random.choice(false_xs)

        return pert



    def randomize(self, v):
        return self.perturbe(self.encode(v))


class Encoding_agg():
    def __init__(self, e, d):
        self.e = e
        self.d = d
        self.p = math.exp(self.e) / (math.exp(self.e) + self.d - 1)
        self.q = 1 / (math.exp(self.e) + self.d - 1)

    def aggregate(self, config):
        reported_values = config['reported_values']
        e = config['epsilon']
        d = config['d']

        results = np.zeros(d)
        n = len(reported_values)

        p = math.exp(e) / (math.exp(e) + d - 1)
        q = 1 / (math.exp(e) + d - 1)

        for i in range(d):
            sum_v = 0
            for j in reported_values:
                if j == i:
                    sum_v += 1
            results[i] = ((sum_v) - n * q) / (p - q)
            if (results[i] < 0):
                results[i] = 0

        return results

