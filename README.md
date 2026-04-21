# LeNet on FashionMNIST
使用 PyTorch 实现的 LeNet-5 模型，在 FashionMNIST 数据集上进行图像分类。

# 环境要求
pip install -r requirements.txt

# 运行流程

1.可视化预览数据集
    python plot.py

2.训练模型
    python model_train.py
    训练完成后，最佳模型权重会自动保存至 ./LeNet/best_model.pth

3.测试模型
    python model_test.py

# 项目结构
LeNet/

├── model.py           # LeNet 网络定义

├── model_train.py     # 训练流程

├── model_test.py      # 测试流程

├── plot.py           # 可视化脚本

├── requirements.txt   # 依赖列表

└── README.md         # 项目说明


