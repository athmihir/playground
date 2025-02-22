import torch
import torch.nn as nn
import torch.optim as optim 
from XOR_modified_model import XORMNN

dataset = [
        [torch.tensor([1,0], dtype=torch.float32).cuda(), torch.tensor([1], dtype=torch.float32).cuda()],
        [torch.tensor([0,1], dtype=torch.float32).cuda(), torch.tensor([1], dtype=torch.float32).cuda()],
        [torch.tensor([1,1], dtype=torch.float32).cuda(), torch.tensor([0], dtype=torch.float32).cuda()],
        [torch.tensor([0,0], dtype=torch.float32).cuda(), torch.tensor([0], dtype=torch.float32).cuda()]
    ]

A_1 = []

for data in dataset:
    X,Y = data
    a_1 = [X[0], X[1], X[0]*X[1], X[0]**2, X[1]**2]
    A_1.append([torch.tensor(a_1, dtype=torch.float32).cuda(), Y])

def train():

    model = XORMNN().to('cuda')
    try:
        model.load_state_dict(torch.load('q3_model.pt',weights_only=True))
    except Exception as e:
        print('Failed to load the model from storage')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for j in range(1000):
        for i, data in enumerate(A_1):
            x,y = data
            optimizer.zero_grad()
            output = model(x)
            loss = nn.BCELoss()
            output = loss(output, y)
            output.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'q3_model.pt')
    evaluate(model)
        
@torch.no_grad()
def evaluate(model):
    # Set the model in train mode.
    model.eval()
    correct = 0
    for data in A_1:
        x,y = data
        y_pred = model(x)
        if y == torch.round(y_pred):
            correct += 1
    print(f"accuracy: {correct/4}")
    

def test_output(beta2, alpha2):
    
    sigmoid = nn.Sigmoid()
    for data in A_1:
        X,Y = data
        print(f"X : {X}")
        z_2 = torch.matmul(beta2.T, X) + alpha2
        print(f"z_2 : {z_2}")
        a_2 = sigmoid(z_2)
        print(f"a_2 = {a_1}")
        if Y == torch.round(a_2):
            print('correct')
        else:
            print('failed')

def get_model_weights():
    model = XORMNN().to('cuda')
    try:
        model.load_state_dict(torch.load('q3_model.pt',weights_only=True))
    except Exception as e:
        print('Failed to load the model from storage')
    
    w2,b2 = model.get_weights_and_biases()
    w2 = torch.round(w2/2)
    b2 = torch.round(b2/2)
    print(w2)
    print(b2)
    test_output(w2, b2)
    

if __name__ == "__main__":
    pass
    # train()
    get_model_weights()
    
