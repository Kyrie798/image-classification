from PIL import Image
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, images, label, transform):
        self.images = images
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        img = Image.open(self.images[item])
        label = self.label[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label