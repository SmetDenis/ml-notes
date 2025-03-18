import numpy as np

from framework.Layer import Layer
from framework.Linear import Linear
from framework.Tensor import Tensor
from framework.Sigmoid import Sigmoid
from framework.Tanh import Tanh

class RNNCell(Layer):
    def __init__(self, n_inputs, n_hidden, n_output, activation='sigmoid'):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        if (activation == 'sigmoid'):
            self.activation = Sigmoid()
        elif (activation == 'tanh'):
            self.activation = Tanh()
        else:
            raise Exception("Non-Linearity not found")

        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)

        self.params += self.w_ih.get_params()
        self.params += self.w_hh.get_params()
        self.params += self.w_ho.get_params()

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size,self.n_hidden)),autograd=True)
