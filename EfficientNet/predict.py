import torch
import json
from torchvision import transforms
from PIL import Image
from model import efficientnetv2_s

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

img_size = {"s": [300, 384],  # train_size, val_size
            "m": [384, 480],
            "l": [384, 480]}
num_model = "s"

transform = transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                transforms.CenterCrop(img_size[num_model][1]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

img = Image.open('./test.jpg')
img = transform(img) # [C, H, W]
img = torch.unsqueeze(img, 0) # [B, C, H, W]
img = img.to(device)

with open('./class_indices.json', 'r') as f:
    class_dict = json.load(f)

efficientnetv2_s = efficientnetv2_s(num_classes=5).to(device)

model_weight_path = "./weights/efficientnetv2_s.pth"
efficientnetv2_s.load_state_dict(torch.load(model_weight_path, map_location=device))
efficientnetv2_s.eval()
with torch.no_grad():
    output = torch.squeeze(efficientnetv2_s(img))
    accuracy = torch.softmax(output, 0)
    predict = output.argmax(0).cpu().numpy()

print('class:{}, accuracy:{:.2f}'.format(class_dict[str(predict)], accuracy[predict].item()))