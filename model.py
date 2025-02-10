import torch
import torch.nn as nn
from torchsummary import summary 

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.A = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float32), requires_grad=True)
        self.B = nn.Parameter(torch.randn(out_features, dtype=torch.float32), requires_grad=True)
    
    def forward(self, x):
        return torch.matmul(self.A.T, x) + self.B
    
    def weight(self):
        return self.A.data
    
    def bias(self):
        return self.B.data

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