import torch
import argparse
from torchvision import transforms 
from utils import read_split_data
from my_dataset import MyDataSet
from model import shufflenet_v2_x1_0
import math
import torch.nn as nn
from tqdm import tqdm

def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:{}'.format(device))

    train_images_path, trian_label, val_images_path, val_label = read_split_data(opt.data_path, 0.2)

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

    model = shufflenet_v2_x1_0(num_classes=opt.num_classes).to(device)
    # 导入预训练权重
    if opt.weights != "":
        weights_dict = torch.load(opt.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                                if model.state_dict()[k].numel() == v.numel()}
        print(model.load_state_dict(load_weights_dict, strict=False))
    
    # 是否冻结
    if opt.freeze_layers:
        for name, para in model.named_parameters():
            if 'fc' not in name:
                para.requires_grad = False
    param = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(param, lr=opt.lr, momentum=0.9, weight_decay=4e-5)
    loss_function = nn.CrossEntropyLoss().to(device)
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lf)
    
    best_acc = 0.0
    for epoch in range(opt.epochs):
        model.train()
        train_bar = tqdm(train_dataloader)
        for step, (img, label) in enumerate(train_bar):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            train_bar.desc = 'Epoch:{}/{} Train Loss:{:.3f}'.format(epoch, opt.epochs, loss)
        
        scheduler.step()
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
                torch.save(model.state_dict(), "./weights/ShuffleNet.pth")
            print('Loss:{:.3f} Accuracy:{:.3f}'.format(total_loss, total_correct / len(val_dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--data_path', type=str, default='./data')
    # dowmload pretrained weights from https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    opt = parser.parse_args()
    main(opt)