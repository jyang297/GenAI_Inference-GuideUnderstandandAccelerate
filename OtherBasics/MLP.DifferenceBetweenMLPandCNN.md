# MLP与CNN对比

## Key Difference
|          |        MLP         |         CNN         |
| :------: | :----------------: | :-----------------: |
| 计算方式 | 全连接层（Linear） |   卷积层（Conv）    |
|  感受野  |        全局        | 局部（Kernel size） |
| 参数共享 |         无         |     共享卷积核      |
|  复杂度  |         高         |         低          |
| 使用任务 | 结构化数据，文本。 |    图像，视频。     |

## Code


### MLP
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 全连接层1
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 全连接层2
        self.softmax = nn.Softmax(dim=1)  # 分类任务Softmax

    def forward(self, x):
        x = self.fc1(x)  # 线性变换
        x = self.relu(x)  # 非线性激活
        x = self.fc2(x)  # 输出层
        x = self.softmax(x)  # 分类任务
        return x

# 定义模型
mlp_model = MLP(input_dim=784, hidden_dim=256, output_dim=10)  # 用于 MNIST（28x28 = 784）
print(mlp_model)
```


### CNN
```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 卷积层1
        self.relu = nn.ReLU()  # 激活函数
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 卷积层2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层
        self.fc2 = nn.Linear(128, num_classes)  # 分类层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)  # 卷积
        x = self.relu(x)
        x = self.pool(x)  # 池化
        x = self.conv2(x)  # 卷积
        x = self.relu(x)
        x = self.pool(x)  # 池化
        x = x.view(x.size(0), -1)  # Flatten 展平
        x = self.fc1(x)  # 全连接层
        x = self.relu(x)
        x = self.fc2(x)  # 分类层
        x = self.softmax(x)
        return x

# 定义模型
cnn_model = CNN(num_classes=10)
print(cnn_model)
```

CNN 关键点
- 采用 卷积层（Conv2d） 进行局部特征提取，共享参数，减少计算量
- 使用 池化层（MaxPool2d） 降维，提高计算效率
- 适用于 图像、视频，能够捕捉局部特征（边缘、纹理）

## What happened?
### Computational Method
```python
# MLP
self.fc1 = nn.Linear(input_dim, hidden_dim)

# CNN
self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
```
MLP:
1. 全连接计算：所有输入神经元和所有输出神经元相连
2. 计算复杂度高，适用于 结构化数据

CNN:
1. 局部计算（感受野限制）：卷积核只关注局部区域
2. 参数共享：一个卷积核扫描整个输入，提高计算效率

## the Change of Dimensions
|              |MLP|CNN
| :----------: | :---------------------: | :-----------------------------------------------: |
| Input Layer  | batch_size × input_dim  |      batch_size × channels × height × width       |
| Hidden Layer | batch_size × hidden_dim | batch_size × num_filters × new_height × new_width |
| Output Layer | batch_size × output_dim |batch_size × num_classes|
