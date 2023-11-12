# import 
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import *
from dataloader import *
from sklearn.model_selection import cross_val_score


# 超参数
batach_size = 1
learning_rate = 1e-3
epochs = 3
dropout_rate = 0.3


# 测试GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 准备数据
root_dir = './Brain-Tumor-Classification-DataSet-master/Training'
glioma_tumor_label_dir = 'glioma_tumor'
meningioma_tumor_label_dir = 'meningioma_tumor'
no_tumor_label_dir = 'no_tumor'
pituitary_tumor_label_dir = 'pituitary_tumor'

# 定义transform
# transform = transforms.Compose([transforms.ToTensor()])
    # 转灰度图像
# transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    # 还要对图像进行重采样
transform = transforms.Compose([
                                # transforms.Resize((64, 64)),
                                transforms.Grayscale(num_output_channels=1), 
                                transforms.ToTensor()])

glioma_tumor_dataset = TrainData(root_dir=root_dir,
                                 label_dir=glioma_tumor_label_dir,
                                 transform=transform)
meningioma_tumor_dataset = TrainData(root_dir=root_dir,
                                     label_dir=meningioma_tumor_label_dir,
                                     transform=transform)
no_tumor_dataset = TrainData(root_dir=root_dir,
                             label_dir=no_tumor_label_dir,
                             transform=transform)
pituitary_tumor_dataset = TrainData(root_dir=root_dir,
                                    label_dir=pituitary_tumor_label_dir,
                                    transform=transform)
train_dataset = glioma_tumor_dataset + meningioma_tumor_dataset + no_tumor_dataset + pituitary_tumor_dataset
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batach_size,
                          shuffle=True,
                          num_workers=0,
                          drop_last=False)
# 创建网络
simplenet = Simplenet(dropout_rate=dropout_rate).to(device)
# simplenet = Simplenet(dropout_rate=dropout_rate)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(simplenet.parameters(),lr = learning_rate)


# 创建SummaryWriter来记录TensorBoard日志
writer = SummaryWriter('./tensorboard/')

# 创建一个类别到整数ID的映射字典
class_to_id = {'no_tumor': 0, 'glioma_tumor': 1, 'meningioma_tumor': 2,'pituitary_tumor':3}



# 训练循环
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = torch.tensor([class_to_id[label] for label in labels]).to(device)
        optimizer.zero_grad()
        outputs = simplenet(inputs)

# 处理标签类型的问题。。。
        # labels = labels[0]
        # labels_int = [class_to_id[label] for label in labels]
        # labels = torch.tensor(labels_int)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 记录每个epoch的平均损失到TensorBoard
    writer.add_scalar('Training Loss', running_loss / (i + 1), epoch)

    print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}')

    # 保存模型权重
    # save_path = './output/'
    if (epoch + 1) % 10 == 0:  # 每隔一定的epoch保存一次
        torch.save({
            'epoch': epoch,
            'model_state_dict': simplenet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / (i + 1)
        }, f'./output/model_checkpoint_epoch_{epoch + 1}.pt')

print('Finished')

# 关闭SummaryWriter
writer.close()