import torch

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
x,y=next(iter(test_loader))

print(x.shape)
print(y.shape)