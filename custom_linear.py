import torch
import torch.nn as nn

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