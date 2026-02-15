from vit_cls_model import ViT_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

################################
# 1. 数据准备
################################
def prepare_data(data_path, batch_size=32, img_size=224):
    """准备训练数据"""
    transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, dataset.classes

################################
# 2. 基本训练循环
################################
def train_model():
    # 训练参数
    BATCH_SIZE = 16
    EPOCHS = 50
    IMG_SIZE = 224
    LEARNING_RATE = 3e-4
    DATA_PATH = 'my_data_dir'  
    # 加载数据
    train_loader, classes = prepare_data(DATA_PATH, BATCH_SIZE, IMG_SIZE)
    print(f'Found {len(classes)} classes: {classes}')
    # 创建模型
    model = ViT_model(
        in_channels=3,
        patch_size=16,
        emb_size=768,
        img_size=IMG_SIZE,
        depth=12,
        n_classes=len(classes),
        num_heads=12,
        drop_p=0.1,
        forward_drop_p=0.1
    ).to(device)
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # 训练循环
    print('===============\n开始训练...\n===============\n')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1} 完成 | 平均损失: {epoch_loss:.4f} | 准确率: {epoch_acc:.2f}%')
    # 保存模型
    torch.save(model.state_dict(), 'final_model.pth')
    print('模型已保存为 final_model.pth')
    return model

################################
# 3. 单张图片预测
################################
def predict(model, image_path, classes):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
    print('===============\n预测结果:\n===============')
    print(f'图片: {image_path}')
    print(f'预测类别: {classes[predicted_idx]}')
    print(f'置信度: {confidence:.2%}')

if __name__ == '__main__':
    # 训练模型
    model = train_model()
    # 预测
    classes = ['dress','hat','longsleeve','outwear','pants',
               'shirt','shoes','shorts','skirt','t-shirt'] 
    images = ['test01.jpg','test02.jpg']
    for img in images:
        predict(model, img, classes)