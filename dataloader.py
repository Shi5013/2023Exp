from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class TrainData(Dataset):
    def __init__(self,
                 root_dir,
                 label_dir,
                 transform = None,
                 target_size=(512, 512)):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform
        self.target_size = target_size


    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir

        # 调整图像大小为目标尺寸
        img = img.resize(self.target_size)
        
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.img_path)