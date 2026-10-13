import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FashionCNN
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)


train_loader=DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
test_dataset=datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
test_loader=DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=FashionCNN()
net=net.to(device)
loss_history=[]
acc_history=[]

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(
    net.parameters(),
    lr=0.01
)

num_epochs=5

for epoch in range(num_epochs):
    total_loss=0.0
    total_samples=0
    correct=0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = net(x)
        y_pred=y_hat.argmax(dim=1)
        correct+=(y_pred==y).sum().item()
        loss = loss_fn(y_hat, y)

        loss.backward()

        optimizer.step()

        total_loss+=x.shape[0]*loss.item()
        total_samples+=x.shape[0]
    avg_loss=total_loss/total_samples
    train_acc=correct/total_samples
    loss_history.append(avg_loss)
    acc_history.append(train_acc)
    print(f'Epoch{epoch+1}/{num_epochs},Loss:{avg_loss:.4f},Train acc:{train_acc:.2%}')

epochs=range(1,num_epochs+1)
fig,axes=plt.subplots(1,2)

axes[0].plot(epochs,loss_history)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[1].plot(epochs,acc_history)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Acc')
axes[1].set_title('Training Acc')
plt.tight_layout()
plt.show()
#测试
net.eval()

test_correct=0
test_total=0

with torch.no_grad():
    for x,y in test_loader:
        x=x.to(device)
        y=y.to(device)

        y_hat=net(x)

        y_pred=y_hat.argmax(dim=1)
        test_correct+=(y_pred==y).sum().item()
        test_total+=x.shape[0]

    test_acc=test_correct/test_total
print(test_acc)
torch.save(net.state_dict(),'fashion-mnist-weight.pth')








