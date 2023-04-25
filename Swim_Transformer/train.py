import os
import torch
import argparse
from torchvision import transforms 
from utils import read_split_data
from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224
import torch.nn as nn
from tqdm import tqdm

def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:{}'.format(device))

    train_images_path, trian_label, val_images_path, val_label = read_split_data(opt.data_path, 0.2)

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    data_transform = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                      'val': transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_dataset = MyDataSet(train_images_path, trian_label, data_transform['train'])
    val_dataset = MyDataSet(val_images_path, val_label, data_transform['val'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    model = swin_tiny_patch4_window7_224(num_classes=opt.num_classes).to(device)
    if opt.weights != "":
        weights_dict = torch.load(opt.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)

    if opt.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    param = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param, lr=opt.lr, weight_decay=5e-2)
    loss_function = nn.CrossEntropyLoss().to(device)
    
    best_acc = 0.0
    for epoch in range(opt.epochs):
        model.train()
        train_bar = tqdm(train_dataloader)
        for img, label in train_bar:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            train_bar.desc = 'Epoch:{}/{} Train Loss:{:.3f}'.format(epoch, opt.epochs, loss)
        
        total_loss = 0.0
        total_correct = 0.0
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(val_dataloader)
            for img, label in val_bar:
                img = img.to(device)
                label = label.to(device)
                output = model(img)
                loss = loss_function(output, label)
                total_loss += loss.item()

                correct = (output.argmax(1) == label).sum()
                total_correct += correct.item()
                val_bar.desc = 'Epoch:{}/{} Val Loss:{:.3f}'.format(epoch, opt.epochs, loss)
            accuracy = total_correct / len(val_dataset)
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), "./weights/swin_tiny.pth")
            print('Loss:{:.3f} Accuracy:{:.3f}'.format(total_loss, total_correct / len(val_dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth')
    parser.add_argument('--freeze_layers', type=bool, default=True)
    opt = parser.parse_args()
    main(opt)