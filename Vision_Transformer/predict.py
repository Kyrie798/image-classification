import torch
import json
from torchvision import transforms
from PIL import Image
from model import vit_base_patch16_224_in21k

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

vit = vit_base_patch16_224_in21k(num_classes=5, has_logits=False).to(device)

model_weight_path = "./weights/vit_base_16.pth"
vit.load_state_dict(torch.load(model_weight_path, map_location=device))
vit.eval()
with torch.no_grad():
    output = torch.squeeze(vit(img))
    accuracy = torch.softmax(output, 0)
    predict = output.argmax(0).cpu().numpy()

print('class:{}, accuracy:{:.2f}'.format(class_dict[str(predict)], accuracy[predict].item()))