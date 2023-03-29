import torch
import json
from torchvision import transforms
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img = Image.open('./test.jpg')
img = transform(img) # [C, H, W]
img = torch.unsqueeze(img, 0) # [B, C, H, W]
img = img.to(device)

with open('./class_indices.json', 'r') as f:
    class_dict = json.load(f)

AlexNet = torch.load('vgg.pth')

AlexNet.eval()
with torch.no_grad():
    output = torch.squeeze(AlexNet(img))
    accuracy = torch.softmax(output, 0)
    predict = output.argmax(0).cpu().numpy()

print('class:{}, accuracy:{:.2f}'.format(class_dict[str(predict)], accuracy[predict].item()))