import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

image = Image.open("test01.jpg")
width, height = image.size
grid_size = 14

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.imshow(image)
patch_w = width // grid_size
patch_h = height // grid_size

for i in range(grid_size + 1):
    ax.axvline(i * patch_w, color='red', linewidth=1, alpha=0.7)
    ax.axhline(i * patch_h, color='red', linewidth=1, alpha=0.7)

for row in range(grid_size):
    for col in range(grid_size):
        center_x = (col + 0.5) * patch_w
        center_y = (row + 0.5) * patch_h
        patch_num = row * grid_size + col
        ax.text(center_x, center_y, f"{patch_num}",
                ha='center', va='center', color='white',
                fontsize=12, weight='bold',
                bbox=dict(boxstyle="circle,pad=0.4",
                          facecolor="blue", alpha=0.9))
ax.set_title(
    f"共{grid_size*grid_size}个空间块", fontsize=14)
ax.axis('off')

plt.tight_layout()
plt.savefig('grid.png', dpi=150, bbox_inches='tight')
plt.show()
