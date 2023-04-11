import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from model import MobileNetV3_Large
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

data_transform = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                  'val': transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_dataset = datasets.ImageFolder(root='./dataset/train', transform=data_transform['train'])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

val_dataset = datasets.ImageFolder(root='./dataset/val', transform=data_transform['val'])
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

MobileNetV3_Large = MobileNetV3_Large(num_classes=5)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(MobileNetV3_Large.parameters(), lr=0.0001)

MobileNetV3_Large = MobileNetV3_Large.to(device)
loss_function = loss_function.to(device)

epochs = 10
best_acc = 0.0
for epoch in range(epochs):
    train_bar = tqdm(train_dataloader)
    MobileNetV3_Large.train()
    for step, (img, label) in enumerate(train_bar):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = MobileNetV3_Large(img)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        train_bar.desc = 'Epoch:{}/{} Train Loss:{:.3f}'.format(epoch, epochs, loss)

    total_loss = 0.0
    total_correct = 0.0
    MobileNetV3_Large.eval()
    with torch.no_grad():
        val_bar = tqdm(val_dataloader)
        for img, label in val_bar:
            img = img.to(device)
            label = label.to(device)

            output = MobileNetV3_Large(img)
            loss = loss_function(output, label)
            total_loss += loss.item()

            correct = (output.argmax(1) == label).sum()
            total_correct += correct.item()
            val_bar.desc = 'Epoch:{}/{} Val Loss:{:.3f}'.format(epoch, epochs, loss)
        accuracy = total_correct / len(val_dataset)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(MobileNetV3_Large, 'MobileNetV3_Large.pth')

        print('Loss:{:.3f} Accuracy:{:.3f}'.format(total_loss, total_correct / len(val_dataset)))
