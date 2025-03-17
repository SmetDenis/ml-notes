import numpy as np

class SGD(object):
    def __init__(self, params, alpha=0.01):
        self.params = params
        self.alpha = alpha

    def step(self, zero=True):
        for weight in self.params:
            weight.data -= weight.grad.data * self.alpha
            if (zero):
                weight.grad.data *= 0

    def zero(self):
        for weight in self.params:
            weight.grad.data *= 0
