import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4*4*64, 512)
        self.fc2 = nn.Linear(512, 10)
 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
# model = torch.load("mnist_cnn.pth")
model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
summary(model, input_size=(1, 28, 28))
model = model.to(device)
traced_script_module = torch.jit.trace(model, torch.ones(1, 1, 28, 28).to(device))
traced_script_module.save("mnist_cnn_cc1.pt")