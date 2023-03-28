import torch
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck')

img = Image.open('./test.jpg')
img = transform(img) # [C, H, W]
img = torch.unsqueeze(img, dim=0) # [B, C, H, W]
img = img.to(device)

LeNet = torch.load('LeNet.pth')

LeNet.eval()
with torch.no_grad():
    output = torch.squeeze(LeNet(img))
    accuracy = torch.softmax(output, 0)
    predict = output.argmax(0).cpu().numpy()

print('class:{}, accuracy:{:.2f}'.format(classes[int(predict)], accuracy[predict].item()))
