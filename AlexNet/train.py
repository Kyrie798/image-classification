import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from model import AlexNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

data_transform = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                  'val': transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

train_dataset = datasets.ImageFolder(root='./dataset/train', transform=data_transform['train'])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

val_dataset = datasets.ImageFolder(root='./dataset/val', transform=data_transform['val'])
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

AlexNet = AlexNet(num_classes=5)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(AlexNet.parameters(), lr=0.0002)

AlexNet = AlexNet.to(device)
loss_function = loss_function.to(device)

epochs = 10
best_acc = 0.0
for epoch in range(epochs):
    print('Epoch:{} / {}'.format(epoch, epochs))
    AlexNet.train()
    for step, (img, label) in enumerate(train_dataloader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = AlexNet(img)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Iterations:{}, Loss:{:.3f}'.format(step, loss.item()))

    total_loss = 0.0
    total_correct = 0.0
    AlexNet.eval()
    with torch.no_grad():
        for img, label in val_dataloader:
            img = img.to(device)
            label = label.to(device)

            output = AlexNet(img)
            loss = loss_function(output, label)
            total_loss += loss.item()

            correct = (output.argmax(1) == label).sum()
            total_correct += correct.item()

        accuracy = total_correct / len(val_dataset)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(AlexNet, 'AlexNet.pth')

        print('Loss:{:.3f}'.format(total_loss))
        print('Accuracy:{:.3f}'.format(total_correct / len(val_dataset)))