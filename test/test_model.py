import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import torch
from modeldefine import ScreenActionNet

# 加载模型
model = ScreenActionNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

# 测试图片文件夹
test_folder = os.path.join(os.path.dirname(__file__), '.')
# 支持多个测试图片
test_images = [f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')]

for img_name in test_images:
    img_path = os.path.join(test_folder, img_name)
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (200, 100))
    state = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(state)
        pred = logits.argmax(dim=1).item()
        print(f"图片 {img_name} 预测区域: {['A', 'B', 'C', 'D'][pred]}")
