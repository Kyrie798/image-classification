import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from model import resnet34
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

data_transform = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                  'val': transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_dataset = datasets.ImageFolder(root='./dataset/train', transform=data_transform['train'])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

val_dataset = datasets.ImageFolder(root='./dataset/val', transform=data_transform['val'])
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

resnet34 = resnet34()
# download resnet34 prd_trained weights from https://download.pytorch.org/models/resnet34-333f7ec4.pth
weight = './resnet34_pre.pth'
resnet34.load_state_dict(torch.load(weight, map_location='cpu'))
in_channel = resnet34.fc.in_features
resnet34.fc = nn.Linear(in_channel, 5)

loss_function = nn.CrossEntropyLoss()

params = [p for p in resnet34.parameters() if p.requires_grad]
optimizer = optim.Adam(resnet34.parameters(), lr=0.0001)

resnet34 = resnet34.to(device)
loss_function = loss_function.to(device)

epochs = 5
best_acc = 0.0
for epoch in range(epochs):
    resnet34.train()
    train_bar = tqdm(train_dataloader)
    for step, (img, label) in enumerate(train_bar):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = resnet34(img)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        train_bar.desc = 'Epoch:{}/{} Train Loss:{:.3f}'.format(epoch, epochs, loss)

    total_loss = 0.0
    total_correct = 0.0
    resnet34.eval()
    with torch.no_grad():
        val_bar = tqdm(val_dataloader)
        for img, label in val_bar:
            img = img.to(device)
            label = label.to(device)

            output = resnet34(img)
            loss = loss_function(output, label)
            total_loss += loss.item()

            correct = (output.argmax(1) == label).sum()
            total_correct += correct.item()
            val_bar.desc = 'Epoch:{}/{} Val Loss:{:.3f}'.format(epoch, epochs, loss)
        accuracy = total_correct / len(val_dataset)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(resnet34, 'resnet34.pth')
        print('Loss:{:.3f} Accuracy:{:.3f}'.format(total_loss, total_correct / len(val_dataset)))
