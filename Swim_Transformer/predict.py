import torch
import json
from torchvision import transforms
from PIL import Image
from model import swin_tiny_patch4_window7_224

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

img = Image.open('./test.jpg')
img = transform(img) # [C, H, W]
img = torch.unsqueeze(img, 0) # [B, C, H, W]
img = img.to(device)

with open('./class_indices.json', 'r') as f:
    class_dict = json.load(f)

swin_tiny = swin_tiny_patch4_window7_224(num_classes=5).to(device)

model_weight_path = "./weights/swin_tiny.pth"
swin_tiny.load_state_dict(torch.load(model_weight_path, map_location=device))
swin_tiny.eval()
with torch.no_grad():
    output = torch.squeeze(swin_tiny(img))
    accuracy = torch.softmax(output, 0)
    predict = output.argmax(0).cpu().numpy()

print('class:{}, accuracy:{:.2f}'.format(class_dict[str(predict)], accuracy[predict].item()))