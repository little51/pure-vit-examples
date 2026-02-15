import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 导入模型定义
from detr_model import DETR, postprocess

# COCO类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 配置参数
model_path = "final_detr_model.pth"
conf_thresh = 0.6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载模型
model = DETR(num_classes=80, num_queries=100).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def detect_image(image_path):
    # 1. 预处理图像
    img = Image.open(image_path).convert('RGB')
    orig_img = img.copy()
    
    transform = Compose([
        Resize((640, 640)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 2. 目标检测
    with torch.no_grad():
        pred_logits, pred_boxes = model(img_tensor)
        boxes, labels, scores = postprocess(pred_logits, pred_boxes, conf_thresh, 640)
    
    # 3. 可视化
    draw = ImageDraw.Draw(orig_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    print(f"\n检测到 {len(boxes)} 个目标:")
    print("-" * 50)
    
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy()
        label = labels[i].item()
        score = scores[i].item()
        
        class_name = COCO_CLASSES[label]
        
        print(f"目标 {i+1}: {class_name} | 置信度: {score:.3f}")
        
        # 绘制边界框（红色）
        draw.rectangle(box.tolist(), outline=(255, 0, 0) , width=3)
        
        # 绘制标签（红色背景，白色文字）
        text = f"{class_name}: {score:.2f}"
        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        draw.rectangle([box[0], box[1]-text_bbox[3]+text_bbox[1]-5, 
                       box[0]+text_bbox[2]-text_bbox[0]+10, box[1]], fill=(255, 0, 0))
        draw.text((box[0]+5, box[1]-text_bbox[3]+text_bbox[1]-5), text, fill=(255,255,255), font=font)
    
    # 4. 显示和保存
    plt.figure(figsize=(12, 8))
    plt.imshow(orig_img)
    plt.axis('off')
    plt.show()
    
    # 生成输出文件名
    output_path = image_path.rsplit('.', 1)[0] + '_detected.jpg'
    orig_img.save(output_path)
    print(f"\n结果已保存到: {output_path}")
    return boxes, labels, scores
    

if __name__ == "__main__":
    detect_image("test03.jpg")
    detect_image("test04.jpg")
    detect_image("test05.jpg")