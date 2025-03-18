import numpy as np

class Tensor(object):
    def __init__(self, data, autograd=False, parents=None, creation_op=None, id=None):
        self.data = np.array(data)
        self.autograd = autograd
        self.parents = parents
        self.children = {}
        self.creation_op = creation_op
        self.grad = None
        if (id is None): id = np.random.randint(100000)
        self.id = id

        if (parents is not None):
            for parent in parents:
                if (self.id not in parent.children):
                    parent.children[self] = 1
                else:
                    parent.children[self] += 1

    def all_grads_propagated(self):
        for _, grads_count in self.children.items():
            if (grads_count != 0): return False
        return True

    def __add__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          parents=[self, other],
                          creation_op="+")
        return Tensor(self.data + other.data)

    def sigmoid(self):
        if (self.autograd):
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          parents=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if (self.autograd):
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          parents=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def __sub__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          parents=[self, other],
                          creation_op="-")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          parents=[self, other],
                          creation_op="*")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if (self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          parents=[self],
                          creation_op="sum_" + str(dim))
        return Tensor(self.data.sum(dim))

    def __neg__(self):
        if (self.autograd):
            return Tensor(self.data * -1,
                          autograd=True,
                          parents=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)

    def index_select(self, indices):
        if (self.autograd):
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         parents=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp, axis=len(self.data.shape)-1, keepdims=True)

        t =  target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)

        target_dist = np.eye(p.shape[1])[t] # One-hot encoding для целевых индексов
        loss = - (np.log(p) * (target_dist)).sum(1).mean()  # Избегаем log(0)

        if (self.autograd):
            out = Tensor(loss,
                         autograd=True,
                         parents=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out
        return Tensor(loss)

    def __repr__(self):
        return str('Tensor(' + self.id.__repr__() + ') Data: ' + self.data.__str__())

    def __str__(self):
        return self.__repr__()

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))

        new_shape = list(self.data.shape) + [copies]

        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if (self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          parents=[self],
                          creation_op="expand_" + str(dim))
        return Tensor(new_data)

    def transpose(self):
        if (self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          parents=[self],
                          creation_op="T")
        return Tensor(self.data.transpose())

    def __matmul__(self, other):
        if (self.autograd):
            return Tensor(self.data @ other.data,
                          autograd=True,
                          parents=[self, other],
                          creation_op="mm")
        return Tensor(self.data @ other.data)

    def mm(self, other):
        return self.__matmul__(other)

    def backward(self, grad=None, grad_origin=None):
        if (self.autograd):
            if (grad == None):
                grad = Tensor(np.ones_like(self.data))

            if (grad_origin is not None):
                if (self.children[grad_origin] == 0):
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin] -= 1

            if (self.grad is None):
                self.grad = grad
            else:
                self.grad += grad

            if ((self.parents is not None) and (self.all_grads_propagated() or grad_origin is None)):
                if (self.creation_op == "+"):
                    self.parents[0].backward(self.grad, grad_origin=self)
                    self.parents[1].backward(self.grad, grad_origin=self)

                if (self.creation_op == "neg"):
                    self.parents[0].backward(self.grad.__neg__())

                if (self.creation_op == '-'):
                    self.parents[0].backward(self.grad, grad_origin=self)
                    self.parents[1].backward(self.grad.__neg__(), grad_origin=self)

                if (self.creation_op == '*'):
                    self.parents[0].backward(self.grad * self.parents[1], grad_origin=self)
                    self.parents[1].backward(self.grad * self.parents[0], grad_origin=self)

                if (self.creation_op == 'mm'):
                    activation = self.parents[0]  # usually an activation function
                    weights = self.parents[1]  # usually a weights matrix
                    activation.backward(self.grad.mm(weights.transpose()))
                    weights.backward(self.grad.transpose().mm(activation).transpose())

                if (self.creation_op == 'T'):
                    self.parents[0].backward(self.grad.transpose())

                if ("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.parents[0].data.shape[dim]
                    self.parents[0].backward(self.grad.expand(dim, ds))

                if ("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.parents[0].backward(self.grad.sum(dim))

                if (self.creation_op == 'sigmoid'):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.parents[0].backward(self.grad * (self * (ones - self)))

                if (self.creation_op == 'tanh'):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.parents[0].backward(self.grad * (ones - (self * self)))

                if self.creation_op == 'index_select':
                    # Инициализация нового градиента с нулями, такой же формы, как у родительского тензора
                    new_grad = np.zeros_like(self.parents[0].data)

                    # Получение выбранных индексов и изменение формы градиента
                    selected_indices = self.index_select_indices.data.flatten()
                    reshaped_grad = grad.data.reshape(len(selected_indices), -1)

                    # Распределение градиентов по выбранным индексам
                    for i in range(len(selected_indices)):
                        new_grad[selected_indices[i]] += reshaped_grad[i]

                    # Обратное распространение градиента к родительскому тензору
                    self.parents[0].backward(Tensor(new_grad))

                if (self.creation_op == 'cross_entropy'):
                    dx = self.softmax_output - self.target_dist
                    self.parents[0].backward(Tensor(dx))

    def forward(self, grad=None, grad_origin=None):
        if (self.autograd):
            if (grad is None):
                grad = Tensor(np.ones_like(self.data))

    def dump(self):
        print("=" * 50)
        print(f"Tensor ID: {self.id}")
        print(f"Data: {self.data}")
        print(f"Autograd: {self.autograd}")
        print(f"Gradient: {self.grad}")
        print(f"Creation Operation: {self.creation_op}")

        print("\nParents:")
        if self.parents is not None:
            for i, parent in enumerate(self.parents):
                print(f"ID: {parent.id}: {parent.data}")
        else:
            print("  None")

        print("\nChildren:")
        if self.children:
            for child_id, count in self.children.items():
                print(f"  Child ID: {child_id.id}, Count: {count}")
        else:
            print("  No children")
