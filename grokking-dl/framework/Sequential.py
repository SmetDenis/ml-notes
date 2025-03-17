from framework.Linear import Linear
from framework.Layer import Layer

class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_params(self):
        params = list()
        for layer in self.layers:
            params += layer.get_params()
        return params
