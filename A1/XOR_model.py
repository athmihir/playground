import torch.nn as nn
from custom_linear import CustomLinear

class XORNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = CustomLinear(2,2)
        self.a1 = nn.Sigmoid()
        self.l2 = CustomLinear(2,2)
        self.a2 = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        return x
    
    def print_weights(self):
        print('weights of first layer:')
        print(self.l1.weight)
        print(self.l1.bias)
        print('weights of second layer are:')
        print(self.l2.weight)
        print(self.l2.bias)

    def get_weights_and_biases(self):
        return((self.l1.weight(), self.l1.bias(), self.l2.weight(), self.l2.bias()))