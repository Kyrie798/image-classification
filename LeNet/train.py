import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from model import LeNet
import torchvision.transforms as transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./root', train=True, download=False, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

val_dataset = torchvision.datasets.CIFAR10(root='./root', train=False, download=False, transform=transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

LeNet = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(LeNet.parameters(), lr=0.001)

LeNet = LeNet.to(device)
loss_function = loss_function.to(device)

epochs = 10
for epoch in range(epochs):
    print('Epoch:{} / {}'.format(epoch, epochs))
    for step, (img, label) in enumerate(train_dataloader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = LeNet(img)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Iterations:{}, Loss:{:.3f}'.format(step, loss.item()))
        
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for img, label in val_dataloader:
            img = img.to(device)
            label = label.to(device)
            
            output = LeNet(img)
            loss = loss_function(output, label) 
            total_loss += loss.item()

            # argmax(1)取output1轴概率最大的索引(网络预测的标签结果)
            # output.argmax(1) == label计算预测结果是否与GT相等
            correct = (output.argmax(1) == label).sum()
            total_correct += correct.item()
        print('Loss:{:.3f}'.format(total_loss))
        print('Accuracy:{:.3f}'.format(total_correct / len(val_dataset)))

torch.save(LeNet, 'LeNet.pth')