import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from pathlib import Path
from vit_detr_model import DETR

BATCH_SIZE = 4           # 批量大小，根据GPU内存调整
NUM_EPOCHS = 100         # 训练轮次

################################
# 1. COCO128数据集加载器
################################
class COCO128Dataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=640):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / 'images' / f'{split}2017'
        self.label_dir = self.root_dir / 'labels' / f'{split}2017'
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')))
        self.transform = Compose([
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert('RGB')
        orig_w, orig_h = img.size
        img_tensor = self.transform(img)
        # 加载标签
        label_path = self.label_dir / f"{self.img_files[idx].stem}.txt"
        boxes, labels = [], []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        cx, cy, w, h = [float(x) for x in data[1:5]]
                        boxes.append([cx, cy, w, h])
                        labels.append(class_id)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        return {
            'image': img_tensor,
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_h, orig_w])
        }

################################
# 2. 匈牙利匹配器
################################
class SimplifiedHungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, nq = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        
        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes'] for t in targets])
        tgt_ids = tgt_ids.long()
        # 分类成本：负对数概率
        class_cost = -out_prob[:, tgt_ids] * self.cost_class
        # L1成本
        bbox_cost = torch.cdist(out_bbox, tgt_bbox, p=1) * self.cost_bbox
        C = class_cost + bbox_cost
        C = C.view(bs, nq, -1).cpu()
        sizes = [len(t['boxes']) for t in targets]
        from scipy.optimize import linear_sum_assignment
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i), torch.as_tensor(j)) for i, j in indices]

################################
# 3. 损失函数
################################
class SimplifiedSetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def forward(self, outputs, targets):
        # 计算匹配
        indices = self.matcher(outputs, targets)
        # 分类损失
        src_logits = outputs['pred_logits']
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                    device=src_logits.device, dtype=torch.long)
        # 将匹配到的目标类别填入
        for i, (src, tgt) in enumerate(indices):
            target_classes[i, src] = targets[i]['labels'][tgt].long()
        # 计算交叉熵损失
        loss_ce = nn.functional.cross_entropy(
            src_logits.flatten(0, 1), 
            target_classes.flatten(0, 1), 
            weight=self.empty_weight
        )
        # 边界框L1损失
        if len(indices) > 0 and all(len(t['boxes']) > 0 for t in targets):
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for src, _ in indices])
            src_boxes = outputs['pred_boxes'][batch_idx, src_idx]
            tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)])
            loss_bbox = nn.functional.l1_loss(src_boxes, tgt_boxes)
        else:
            # 如果没有匹配的目标，bbox损失为0
            loss_bbox = torch.tensor(0.0, device=src_logits.device)
        # 分类和边界框损失
        return {
            'loss_ce': loss_ce * self.weight_dict['loss_ce'], 
            'loss_bbox': loss_bbox * self.weight_dict['loss_bbox']
        }

################################
# 4. 训练函数
################################
def train_epoch(model, criterion, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        pred_logits, pred_boxes = model(images)
        outputs = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f'Epoch {epoch}, Batch {i}/{len(loader)}, Loss: {loss.item():.4f}')
    return total_loss / len(loader)

################################
# 5. 验证函数
################################
@torch.no_grad()
def validate(model, criterion, loader, device):
    model.eval()
    total_loss = 0
    for images, targets in loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        pred_logits, pred_boxes = model(images)
        outputs = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
        loss_dict = criterion(outputs, targets)
        total_loss += sum(loss_dict.values()).item()
    avg_loss = total_loss / len(loader)
    return avg_loss


def collate_fn(batch):
    return torch.stack([b['image'] for b in batch]), [{k: v for k, v in b.items() if k != 'image'} for b in batch]


################################
# 6. 主函数
################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    # 数据加载
    train_loader = DataLoader(
        COCO128Dataset('coco128_yolo', 'train'),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(
        COCO128Dataset('coco128_yolo', 'val'),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f'训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}')
    # 模型
    model = DETR(num_classes=80, num_queries=100,num_heads=8, emb_size=768).to(device)
    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
    # 优化器
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # 损失函数
    matcher = SimplifiedHungarianMatcher(cost_class=1, cost_bbox=5)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
    criterion = SimplifiedSetCriterion(80, matcher, weight_dict, eos_coef=0.1).to(device)
    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_epoch(model, criterion, train_loader, optimizer, device, epoch)
        val_loss = validate(model, criterion, val_loader, device)
        scheduler.step()
        print(f'Epoch {epoch}: 训练损失 {train_loss:.4f}, 验证损失 {val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型')
    torch.save(model.state_dict(), 'final_detr_model.pth')
    print('训练完成！')

if __name__ == '__main__':
    main()