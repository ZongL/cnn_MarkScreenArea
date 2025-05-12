import cv2
import torch
import numpy as np
import os
from modeldefine import ScreenActionNet, ReinforcementLearningAgent

# 初始化模型和强化学习代理
model = ScreenActionNet()
agent = ReinforcementLearningAgent(model)

# 模拟训练
image_folder = "path_to_images"  # 假设图像存储在此文件夹中
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

for episode, image_file in enumerate(image_files):
    # 采集图像
    frame = cv2.imread(os.path.join(image_folder, image_file))
    frame = cv2.resize(frame, (200, 100))  # 调整尺寸
    state = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

    # 模型预测
    action_probs = agent.predict(state)
    action = np.argmax(action_probs)

    # 人工反馈
    print(f"模型预测区域: {['A', 'B', 'C', 'D'][action]}")
    reward = float(input("输入奖励(1为正确，-1为错误): "))

    # 更新模型
    agent.update(state, action, reward)