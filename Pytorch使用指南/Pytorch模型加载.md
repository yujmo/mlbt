# 1 Pytorch模型加载[^1]
## 1.1 保存整个神经网络的结构信息和模型的参数信息,save的对象是网络net

`
torch.save(model_object,'resnet.pth')
model = torch.load('resnet.pth')
`

## 1.2 只保存神经网络的训练模型参数,save的对象是net.state_dict()

`
torch.save(my_resnet.state_dict(),'my_resnet.pth')
my_resnet.load_state_dict(torch.load('my_resnet.pth'))
`

# 2 pytorch预训练模型
## 2.1 加载预训练模型

`
import torchvision.models as models
resnet18=models.resnet18(pretrained=True)
`

## 2.2 只加载模型，不加载预训练的参数

`
resnet18=models.resnet18(pretrained=False)
resnet18.load_state_dict(torch.load('resnet18-5c106cde.pth'))
`

## 2.3 微该基础模型
    
    对于一些任务而言，有些层并不是直接能用的，需要我们进行稍微改一下，比如，resnet最后的全连接层是分1000类，而我们此时只要21类；又比如，resnet第一层卷积层接收的通过是3，我们可以输入图片的通道是4，那么可以通过以下方法进行修改：
`
resnet.conv1=nn.Conv2d(4,64,kernel_size=7,stride=2,padding=3,bias=False)
resnet.fc=nn.Linear(2048,21)
  
resnet=torchvision.models.resnet152(pretrained=True)
resnet.fc=torch.nn.Linear(2048,10)
`


[^1]: https://blog.csdn.net/qq_42698422/article/details/100547225
