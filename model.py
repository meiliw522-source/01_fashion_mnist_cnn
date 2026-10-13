from torch import nn

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(400,64),nn.ReLU(),
            nn.Linear(64,10)
            )
    def forward(self,x):
        return self.net(x)
