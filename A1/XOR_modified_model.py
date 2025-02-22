import torch.nn as nn
from custom_linear import CustomLinear

class XORMNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = CustomLinear(5,1)
        self.a2 = nn.Sigmoid()

    def forward(self, x):
        x = self.l2(x)
        x = self.a2(x)
        return x
    
    def print_weights(self):
        print('weights of second layer are:')
        print(self.l2.weight)
        print(self.l2.bias)

    def get_weights_and_biases(self):
        return((self.l2.weight(), self.l2.bias()))