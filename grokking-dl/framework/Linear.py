from framework.Layer import Layer
from framework.Tensor import Tensor
import numpy as np

class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        self.weight = Tensor(
            np.random.rand(n_inputs, n_outputs) * np.sqrt(2.0 / (n_inputs)),
            autograd=True
        )

        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.params.append(self.weight)
        self.params.append(self.bias)

    def forward(self, input):
        return input @ self.weight + self.bias.expand(0, len(input.data))
