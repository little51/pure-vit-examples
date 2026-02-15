import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor
from einops.layers.torch import Rearrange
from einops import repeat
from PIL import Image

################################
# 1. 图像预处理
################################
img = Image.open('test01.jpg')
transform = Compose([Resize((640, 640)), ToTensor()])
x = transform(img).unsqueeze(0)

################################
# 2. Patch嵌入
################################
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=640):
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = (img_size // patch_size) ** 2
        self.positions = nn.Parameter(torch.randn(num_patches + 1, emb_size))
        
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
    
    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '1 1 e -> b 1 e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        return x + self.positions

################################
# 3. Transformer编码器层
################################
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.proj = nn.Linear(emb_size, emb_size)
        
    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.emb_size // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.emb_size // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, self.emb_size)
        return self.proj(x)

class FeedForward(nn.Module):
    def __init__(self, emb_size, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Linear(expansion * emb_size, emb_size)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadSelfAttention(emb_size, num_heads)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = FeedForward(emb_size)
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self, x): 
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


################################
# 4. Transformer解码器层
################################
class TransformerDecoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.self_attn = MultiHeadSelfAttention(emb_size, num_heads)
        
        self.norm2 = nn.LayerNorm(emb_size)
        self.cross_attn = nn.MultiheadAttention(emb_size, num_heads, batch_first=True)
        
        self.norm3 = nn.LayerNorm(emb_size)
        self.ffn = FeedForward(emb_size)
        
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, query, memory):
        # 自注意力
        q = self.norm1(query)
        query = query + self.dropout(self.self_attn(q))
        
        # 交叉注意力
        q = self.norm2(query)
        attn_out, _ = self.cross_attn(q, memory, memory)
        query = query + self.dropout(attn_out)
        
        # FFN
        q = self.norm3(query)
        query = query + self.dropout(self.ffn(q))
        
        return query

################################
# 5. DETR模型
################################
class DETR(nn.Module):
    def __init__(self, 
                 num_classes=80,           # 类别数
                 num_queries=100,           # 目标查询数
                 emb_size=768,              # 嵌入维度
                 num_heads=8,                # 注意力头数
                 encoder_layers=6,           # 编码器层数
                 decoder_layers=6):          # 解码器层数
        super().__init__()
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(emb_size=emb_size)
        
        # 编码器
        self.encoder = nn.ModuleList([
            TransformerBlock(emb_size, num_heads) for _ in range(encoder_layers)
        ])
        
        # 解码器
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(emb_size, num_heads) for _ in range(decoder_layers)
        ])
        
        # 可学习的目标查询
        self.query_embed = nn.Embedding(num_queries, emb_size)
        
        # 输出头
        self.class_head = nn.Linear(emb_size, num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 4),
            nn.Sigmoid()  # 输出归一化的坐标
        )
        
    def forward(self, x):
        # Patch嵌入
        x = self.patch_embed(x)  # [B, N+1, E]
        
        # 编码器
        for layer in self.encoder:
            x = layer(x)
        memory = x[:, 1:, :]  # [B, N, E] 移除cls_token
        
        # 解码器
        batch_size = x.shape[0]
        query = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        for layer in self.decoder:
            query = layer(query, memory)
        
        # 预测
        class_logits = self.class_head(query)  # [B, Q, C+1]
        bbox_preds = self.bbox_head(query)     # [B, Q, 4]
        
        return class_logits, bbox_preds

################################
# 6. 后处理
################################
def postprocess(pred_logits, pred_boxes, conf_thresh=0.5, img_size=640):
    # 计算置信度
    probs = pred_logits[0].softmax(-1)  # [Q, C+1]
    scores, labels = probs[:, :-1].max(-1)
    # 过滤低置信度
    keep = scores > conf_thresh
    scores = scores[keep]
    labels = labels[keep]
    boxes = pred_boxes[0][keep]  # [N, 4] 格式为 [cx, cy, w, h]，范围 [0,1]
    if len(boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long), torch.zeros(0)
    # 转换坐标格式: [cx, cy, w, h] -> [x1, y1, x2, y2]
    # 先分离坐标分量
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 计算x1, y1, x2, y2
    x1 = (cx - w/2) * img_size
    y1 = (cy - h/2) * img_size
    x2 = (cx + w/2) * img_size
    y2 = (cy + h/2) * img_size
    # 重新组合
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes, labels, scores

################################
# 7. 测试
################################
# 创建模型
model = DETR(num_classes=80, num_queries=100)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 前向传播
class_logits, bbox_preds = model(x)
print(f"类别输出: {class_logits.shape}")  # [1, 100, 81]
print(f"边界框输出: {bbox_preds.shape}")   # [1, 100, 4]

# 后处理
boxes, labels, scores = postprocess(class_logits, bbox_preds, conf_thresh=0.3)
print(f"检测到 {len(boxes)} 个目标")