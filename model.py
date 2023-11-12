# import
import torch
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Sequential,Linear,Flatten,Softmax
from PIL import Image
import torchvision.transforms as transforms

# 定义网络
class Simplenet(nn.Module):
    def __init__(self,dropout_rate):
        super(Simplenet,self).__init__()
        self.model1 = Sequential(
            Conv2d(1,32,3,padding=1,stride=1),
            MaxPool2d(2,stride=2),
            Conv2d(32,64,3,padding=1,stride=1),
            MaxPool2d(2,stride=2),
            Conv2d(64,64,3,padding=1,stride=1),
            MaxPool2d(2,stride=2),
            Conv2d(64,128,3,padding=1,stride=1),
            MaxPool2d(2,stride=2),
            Conv2d(128,256,3,padding=1,stride=1),
            MaxPool2d(2,stride=2),
            # 全连接层
            Flatten(),
            Linear(256 * 16 * 16, 32), # 全连接层1，将输出变为 32
            nn.Dropout(dropout_rate), 
            # Linear(32, 3),  # 全连接层2，将输出变为 3
            Linear(32, 4),  # 全连接层2，将输出变为 4
            Softmax(dim=1)  # softmax 层，执行四分类
        )

    
    def forward(self,x):
        x = self.model1(x)
        return x

"""
# 读取文件 1870✖️1870
image = Image.open("/Users/shi/Project_01/voxelmorph-dev/test/xidian_logo.png")

# 改变大小 64✖️64
image = image.resize((64, 64))

# 将图片转化为灰度图像（如果它还不是灰度的）
image = image.convert("L")

# 将图像数据转化为张量
transform = transforms.ToTensor()
input_data = transform(image).unsqueeze(0)
print(input_data.size())
# 实例化
simplenet = Simplenet()
output = simplenet(input_data)

print(output.size())
print(output)
print(simplenet)
"""