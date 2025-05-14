import cv2
import torch
import numpy as np
import os
from modeldefine import ScreenActionNet, ReinforcementLearningAgent
from torch.utils.tensorboard import SummaryWriter  # 新增

# 初始化模型和强化学习代理
model = ScreenActionNet()
model_path = 'model.pth'
# 如果有已保存的模型参数则加载，实现累积训练
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"已加载模型参数: {model_path}")
else:
    print("未检测到已保存模型，将从头开始训练。")
agent = ReinforcementLearningAgent(model)

writer = SummaryWriter(log_dir="runs/exp1")  # 新增

# 添加网络结构到tensorboard
example_input = torch.zeros(1, 3, 100, 200)
writer.add_graph(model, example_input)

# 模拟训练
image_folder = "path_to_images"  # 假设图像存储在此文件夹中
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

for episode, image_file in enumerate(image_files):
    # 采集图像
    print(f"当前读取图片: {image_file}")  # 打印当前图片文件名
    frame = cv2.imread(os.path.join(image_folder, image_file))
    frame = cv2.resize(frame, (200, 100))  # 调整尺寸
    state = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

    # 模型预测
    action_probs = agent.predict(state)
    action = np.argmax(action_probs)

    # 人工反馈
    print(f"模型预测区域: {['A', 'B', 'C', 'D'][action]}")
    reward = float(input("输入奖励(1为正确，-1为错误): "))

    # 记录reward到tensorboard
    writer.add_scalar('Reward', reward, episode)

    # 记录权重分布到tensorboard
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, episode)

    # 更新模型
    agent.update(state, action, reward)

# 训练结束后保存模型参数
torch.save(model.state_dict(), model_path)
print(f"模型参数已保存到: {model_path}")

writer.close()  # 关闭writer