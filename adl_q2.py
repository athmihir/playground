import torch
import torch.nn as nn
from torchsummary import summary 
import torch.optim as optim 
from XOR_model import XORNN

dataset = [
        [torch.tensor([1,0], dtype=torch.float32).cuda(), torch.tensor([0,1], dtype=torch.float32).cuda()],
        [torch.tensor([0,1], dtype=torch.float32).cuda(), torch.tensor([0,1], dtype=torch.float32).cuda()],
        [torch.tensor([1,1], dtype=torch.float32).cuda(), torch.tensor([1,0], dtype=torch.float32).cuda()],
        [torch.tensor([0,0], dtype=torch.float32).cuda(), torch.tensor([1,0], dtype=torch.float32).cuda()]
    ]

def train():

    model = XORNN().to('cuda')
    try:
        model.load_state_dict(torch.load('model.pt',weights_only=True))
    except Exception as e:
        print('Failed to load the model from storage')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for j in range(1000):
        for i, data in enumerate(dataset):
            x,y = data
            optimizer.zero_grad()
            output = model(x)
            loss = nn.BCELoss()
            output = loss(output, y)
            output.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'model.pt')
    evaluate(model)
        
@torch.no_grad()
def evaluate(model):
    # Set the model in train mode.
    model.eval()
    correct = 0
    for data in dataset:
        x,y = data
        y_pred = model(x)
        if torch.argmax(y) == torch.argmax(y_pred):
            correct += 1
    print(f"accuracy: {correct/4}")
    

def test_output(beta1, alpha1, beta2, alpha2):
    
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=0)
    for index, data in enumerate(dataset):
        X,Y = data
        print(f"X : {X}")
        z_1 = torch.matmul(beta1.T, X) + alpha1
        print(f"z_1 : {z_1}")
        a_1 = sigmoid(z_1)
        print(f"a_1 = {a_1}")
        z_2 = torch.matmul(beta2.T, a_1) + alpha2
        print(f"z_2 = {z_2}")
        a_2 = softmax(z_2)
        if torch.argmax(Y) == torch.argmax(a_2):
            print('correct')
        else:
            print('failed')

def get_model_weights():
    model = XORNN().to('cuda')
    try:
        model.load_state_dict(torch.load('model.pt',weights_only=True))
    except Exception as e:
        print('Failed to load the model from storage')
    
    w1,b1,w2,b2 = model.get_weights_and_biases()
    w1 = torch.round(w1*10/4)
    b1 = torch.round(b1*10/4)
    w2 = torch.round(w2*10/4)
    b2 = torch.round(b2*10/4)
    print(w1)
    print(b1)
    print(w2)
    print(b2)
    test_output(w1, b1, w2, b2)
    

if __name__ == "__main__":
    # train()
    get_model_weights()
    
