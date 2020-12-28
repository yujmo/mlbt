import torch.nn as nn
import torch.nn.functional as F
import torch

"""
    1、生成神经网络模型model
    2、保存model为pth文件
    3、使用netron工具打开pth文件
""" 
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.mp = nn.MaxPool2d(2)
        self.mp1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2560, 512)
        self.fc2 = nn.Linear(512, 10)
 
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp1(self.conv3(x)))
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
model = Net().to(device)

torch.save(model, './model_para.pth')
