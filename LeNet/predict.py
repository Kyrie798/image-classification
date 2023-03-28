import torch
import torchvision.transforms as transforms

from PIL import Image

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:{}'.format(device))

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')

    LeNet = torch.load('LeNet.pth')

    img = Image.open('./test.jpg')
    img = transform(img) # [C, H, W]
    img = torch.unsqueeze(img, dim=0) # [B, C, H, W]
    img = img.to(device)
    with torch.no_grad():
        output = LeNet(img)
        predict = output.argmax(1)

    print(classes[int(predict)])

if __name__ == '__main__':
    main()