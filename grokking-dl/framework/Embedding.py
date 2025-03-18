from framework.Layer import Layer
from framework.Tensor import Tensor
import numpy as np

class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        # этот стиль инициализации является соглашением от word2vec
        weight = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(weight, autograd=True)
        self.params.append(self.weight)

    def forward(self, input):
        # input is word indices
        return self.weight.index_select(input)
