import torch
import torch.nn as nn
import torch.nn.functional as F

class ScreenActionNet(nn.Module):
    def __init__(self):
        super(ScreenActionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 池化层

        # 动态计算全连接层输入大小
        dummy_input = torch.zeros(1, 3, 100, 200)  # 假设输入图像大小为 (3, 100, 200)
        conv_output_size = self._get_conv_output_size(dummy_input)

        self.fc1 = nn.Linear(conv_output_size, 128)  # 动态调整输入大小
        self.fc2 = nn.Linear(128, 4)  # 输出4个区域的概率

    def _get_conv_output_size(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 直接输出logits，不加softmax

class ReinforcementLearningAgent:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失

    def predict(self, state):
        with torch.no_grad():
            logits = self.model(state)
            probs = F.softmax(logits, dim=1)
            return probs.numpy()

    def update(self, state, action, reward):
        # 只在reward为正时训练，reward为负时跳过
        if reward > 0:
            logits = self.model(state)
            target = torch.tensor([action], dtype=torch.long)
            loss = self.criterion(logits, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # reward为负时不做任何更新
