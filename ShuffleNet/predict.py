import torch
import json
from torchvision import transforms
from PIL import Image
from model import shufflenet_v2_x1_0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = Image.open('./test.jpg')
img = transform(img) # [C, H, W]
img = torch.unsqueeze(img, 0) # [B, C, H, W]
img = img.to(device)

with open('./class_indices.json', 'r') as f:
    class_dict = json.load(f)

shufflenet_v2 = shufflenet_v2_x1_0(num_classes=5).to(device)

model_weight_path = "./weights/ShuffleNet.pth"
shufflenet_v2.load_state_dict(torch.load(model_weight_path, map_location=device))
shufflenet_v2.eval()
with torch.no_grad():
    output = torch.squeeze(shufflenet_v2(img))
    accuracy = torch.softmax(output, 0)
    predict = output.argmax(0).cpu().numpy()

print('class:{}, accuracy:{:.2f}'.format(class_dict[str(predict)], accuracy[predict].item()))