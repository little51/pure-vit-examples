# 纯Vision Transformer实现CV的例子

## 一、建立虚拟环境

```shell
# 创建虚拟环境
conda create -n purevit python=3.13 -y
# 激活虚拟环境
conda activate purevit
# 安装依赖库
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 验证PyTorch是否安装成功
python -c "import torch; print(torch.cuda.is_available())"
```

## 二、图像分类

```shell
# 1、激活虚拟环境
conda activate purevit
# 2、模型结构
python vit_cls_model.py
# 3、训练数据准备
git clone https://github.com/lightly-ai/dataset_clothing_images.git my_data_dir
#  删除数据目录下的.git目录
#  Linux执行以下命令，Windows上直接删除.git目录
rm -rf my_data_dir/.git
# 4、模型训练
python vit_cls_train.py
# 5、分类推理
python vit_cls_pred.py
```

## 三、目标检测

```shell
# 1、激活虚拟环境
conda activate purevit
# 2、模型结构
python vit_detr_model.py
# 3、训练数据准备
https://github.com/lightly-ai/coco128_yolo/releases/download/v0.0.1/coco128_yolo.zip
# 4、模型训练
python vit_detr_train.py
# 5、目标检测推理
python vit_detr_pred.py
```

