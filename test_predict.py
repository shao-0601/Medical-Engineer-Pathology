import torch
import torch.nn as nn
import torch.optim as optim
from medmnist import PathMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 解决中文显示和字体警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义简单CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # PathMNIST是9分类，28x28经过两次池化后是7x7
        self.fc_layers = nn.Linear(32 * 7 * 7, 9)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 加载数据集（带ToTensor转换）
transform = transforms.ToTensor()
train_dataset = PathMNIST(split='train', download=True, transform=transform)
test_dataset = PathMNIST(split='test', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数、优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
print("开始训练模型...")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        # 把标签从[N,1]变成[N]适配损失函数
        loss = criterion(outputs, labels.squeeze().long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/5, 平均损失: {running_loss/len(train_loader):.4f}")

# 测试集计算准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

acc = 100 * correct / total
print(f"\n测试集准确率: {acc:.2f}%")

# 可视化单张测试图片（带真实类别、预测类别、准确率）
img, label = test_dataset[0]
model.eval()
with torch.no_grad():
    output = model(img.unsqueeze(0))  # 增加batch维度
    _, predicted = torch.max(output, 1)

# 转换为matplotlib可显示的格式（CHW→HWC）
img_show = img.permute(1, 2, 0).numpy()

plt.figure(figsize=(4, 4))
plt.imshow(img_show)
plt.title(f"真实类别: {label[0]}\n预测类别: {predicted.item()}\n准确率: {acc:.2f}%")
plt.axis('off')
plt.savefig("test_sample_with_pred.png", dpi=150)  # 高清保存
plt.show()

print("\n✅ 带预测结果的图片已保存为: test_sample_with_pred.png")