# 导入测试所需的库
from train import *
from model import *
from dataloader import *
from sklearn.metrics import accuracy_score, classification_report




best_model = Simplenet(dropout_rate=dropout_rate).to(device)
best_model_path = './output/model_checkpoint_epoch_200.pt'  # 填写保存最佳模型的路径
checkpoint = torch.load(best_model_path)
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model.eval()
correct = 0
total = 0

# 准备测试数据集
root_dir = './Brain-Tumor-Classification-DataSet-master/Testing'
glioma_tumor_label_dir = 'glioma_tumor'
meningioma_tumor_label_dir = 'meningioma_tumor'
no_tumor_label_dir = 'no_tumor'
pituitary_tumor_label_dir = 'pituitary_tumor'

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
test_dataset = glioma_tumor_dataset + meningioma_tumor_dataset + no_tumor_dataset + pituitary_tumor_dataset
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batach_size,
                          shuffle=False,
                          num_workers=0,
                          drop_last=False)

# 存储预测标签和真实标签
all_predictions = []
all_labels = []

# 不进行梯度计算

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = torch.tensor([class_to_id[label] for label in labels]).to(device)
        outputs = best_model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
