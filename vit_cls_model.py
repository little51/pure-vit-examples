from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor
from einops.layers.torch import Rearrange,Reduce    
from einops import repeat
import torch
from torch import nn,Tensor
from torchsummary import summary

################################
# 1. 读取图像并显示
################################
img = Image.open('test01.jpg')
fig = plt.figure()
plt.imshow(img)
plt.show()

################################
# 2. 图像预处理成224x224的张量
################################
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
print("Shape of the image tensor:", x.shape)
x = x.unsqueeze(0)
print("Shape of the batched tensor:", x.shape)


################################
# 3. PatchEmbedding
################################
class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 patch_size: int = 16, 
                 emb_size: int = 768, 
                 img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        # 生成一个维度为emb_size的向量当做cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # 位置编码信息，一共有197个位置向量
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, emb_size))

        self.projection = nn.Sequential(
            # 1 3 (224 224)  -> 1 (224/16 224/16) (16*16*3) = 1 196 768
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            # Linear层将每个patch映射到emb_size维的空间，维度依然为1 196 768
            nn.Linear(patch_size * patch_size * in_channels, emb_size),
        )
                
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x
        
patches_embedded = PatchEmbedding()(x)
print("Shape of the embedded patches tensor:", patches_embedded.shape)

################################
# 4. MultiHeadSelfAttention
################################
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads=12, dropout=0., **kwargs):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, **kwargs)
        
    def forward(self, x):
        # 调整维度顺序为 [seq_len, batch, emb_size]
        x = x.transpose(0, 1)
        x, _ = self.attention(x, x, x)
        return x.transpose(0, 1)

attention = MultiHeadSelfAttention(emb_size=768)(patches_embedded)
print("Shape of the attention output tensor:", attention.shape)

################################
# 5. ResidualAdd
################################
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
################################
# 6. FeedForwardBlock
################################
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


################################
# 7. TransformerEncoderBlock
################################
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 num_heads: int = 12,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadSelfAttention(emb_size, num_heads=num_heads, dropout=drop_p, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
        ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

transformer_encode = TransformerEncoder()(patches_embedded)
print("Shape of the transformer output tensor:", transformer_encode.shape)

################################
# 8. ClassificationHead
################################
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
        
################################
# 9. ViT Model
################################
class ViT_model(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

################################
# 10. 测试ViT模型
################################
model = ViT_model(n_classes=10)
print("Stuct of the ViT model:\n")
summary(model, input_size=[(3, 224, 224)], batch_size=1, device="cpu")
output = model(x)
print("Output shape:", output.shape)
predictions = torch.softmax(output, dim=1)
predicted_class = torch.argmax(predictions, dim=1)
print("Predicted class:", predicted_class.item())