from vit_cls_model import ViT_model
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

def load_model(model_path, num_classes):
    """加载训练好的模型"""
    model = ViT_model(
        in_channels=3,
        patch_size=16,
        emb_size=768,
        img_size=224,
        depth=12,
        n_classes=num_classes,
        num_heads=12,
        drop_p=0.1,
        forward_drop_p=0.1
    ).to(device)
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, image_path, classes):
    """预测单张图片"""
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
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
    model_path = 'final_model.pth'
    model = load_model(model_path, 10)
    classes = ['dress','hat','longsleeve','outwear','pants',
               'shirt','shoes','shorts','skirt','t-shirt'] 
    images = ['test01.jpg','test02.jpg']
    for img in images:
        predict(model, img, classes)