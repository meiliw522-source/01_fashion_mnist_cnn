import torch
from torch import nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset=datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader=DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader=DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

net=nn.Sequential(
    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.ReLU(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(400,64),nn.ReLU(),
    nn.Linear(64,10)
)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

for x,y in train_loader:
    x=x.to(device)
    y=y.to(device)

    y_hat=net(x)

