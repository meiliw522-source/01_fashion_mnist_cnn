import torch
from torchvision import datasets,transforms
from model import  FashionCNN

transform=transforms.ToTensor()

test_dataset=datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=FashionCNN().to(device)
#加载参数
state_dict=torch.load(
    'fashion-mnist-weight.pth',
    map_location=device
)

net.load_state_dict(state_dict)
net.to(device)

net.eval()
x,y=test_dataset[0]
x=x.unsqueeze(0)

with torch.no_grad():
    x=x.to(device)
    y_hat=net(x)
    y_hat=torch.softmax(y_hat,dim=1)
    y_pred=y_hat.argmax(dim=1).item()
    print(f'预测类型:{test_dataset.classes[y_pred]}')
    print(f'真实类型:{test_dataset.classes[y]}')
    print(f'预测概率:{y_hat.max().item():.2%}')