import cv2
import torch
import numpy as np
import os
from modeldefine import ScreenActionNet
from torch.utils.tensorboard import SummaryWriter

# 初始化模型
model = ScreenActionNet()
model_path = 'model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"已加载模型参数: {model_path}")
else:
    print("未检测到已保存模型，将从头开始训练。")

writer = SummaryWriter(log_dir="runs/exp1")

example_input = torch.zeros(1, 3, 100, 200)
writer.add_graph(model, example_input)

image_folder = "path_to_images"
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 标签映射 1.jpg->0, 2.jpg->1, 3.jpg->2, 4.jpg->3
label_map = {str(i+1)+'.jpg': i for i in range(4)}

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
epochs = 1000  # 训练轮数，可根据需要调整
episode = 0

try:
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for image_file in image_files:
            label = label_map[image_file]
            frame = cv2.imread(os.path.join(image_folder, image_file))
            frame = cv2.resize(frame, (200, 100))
            state = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
            target = torch.tensor([label], dtype=torch.long)

            logits = model(state)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1).item()
            if pred == label:
                correct += 1
            writer.add_scalar('Loss', loss.item(), episode)
            writer.add_scalar('Acc', int(pred == label), episode)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, episode)
            episode += 1
        acc = correct / len(image_files)
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Acc={acc:.2f}")
        if acc == 1.0:
            print("已全部正确分类，提前停止训练。")
            break
except KeyboardInterrupt:
    print("\n训练被手动中断，正在保存模型参数...")
finally:
    torch.save(model.state_dict(), model_path)
    print(f"模型参数已保存到: {model_path}")
    writer.close()

# 推理函数示例
def predict_image(image_path):
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (200, 100))
    state = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(state)
        pred = logits.argmax(dim=1).item()
        return ['A', 'B', 'C', 'D'][pred]

# 用法示例：
# print(predict_image('path_to_images/1.jpg'))