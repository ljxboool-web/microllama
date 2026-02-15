## 环境配置
### 1. 系统要求
| 项目 | 要求 | 当前配置 |
|------|------|--------|
| Python | >= 3.8 | 3.12 |
| CUDA | 可选 (10.2+) | 12.2 |
| 内存 | >= 8GB | 充足 |
| 硬盘 | >= 20GB | 充足 |
### 2. 必需的Python包
```
PyTorch >= 1.9.0
DGL >= 0.6.0
scikit-learn >= 0.24
numpy >= 1.19.0
scipy >= 1.5.0
tqdm >= 4.50.0
networkx >= 2.5
```
### 3. 安装步骤
#### 方法 A: 使用离线轮子文件（推荐用于服务器）

```bash
# 1. 进入项目目录
cd ~/LIP_MLNC

# 2. 进入轮子文件目录
cd wheels

# 3. 使用自动脚本安装（推荐）
bash ../install_final.sh

# 或者手动安装（逐步）
python3.12 -m pip install --no-index --find-links ./ --break-system-packages numpy scipy scikit-learn -q
python3.12 -m pip install --no-index --find-links ./ --break-system-packages torch torchvision torchaudio -q
python3.12 -m pip install --no-index --find-links ./ --break-system-packages dgl networkx tqdm -q
```

#### 方法 B: 使用在线pip安装（本地开发）

```bash
# 使用 conda 环境（推荐）
conda create -n lip_mlnc python=3.12
conda activate lip_mlnc

# 或使用虚拟环境
python3.12 -m venv venv
source venv/bin/activate

# 安装依赖
pip install torch torchvision torchaudio dgl scikit-learn networkx tqdm numpy scipy
```

### 4. 验证安装

```bash
python3.12 << 'EOF'
import torch
import dgl
import sklearn
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ DGL: {dgl.__version__}')
print(f'✓ scikit-learn: {sklearn.__version__}')
print(f'✓ CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU 数量: {torch.cuda.device_count()}')
EOF
```

---

## 数据准备

### 1. 支持的数据集

项目支持以下数据集:

| 数据集 | 节点数 | 边数 | 标签数 | 位置 |
|--------|--------|------|--------|------|
| DBLP | 28,702 | 136,670 | 4 | `mlncData/dblp/` |
| BlogCatalog | ~10,312 | ~333,983 | ~39 | `mlncData/blogcatalog/` |
| PCG | ~3,890 | ~13,316 | ~50 | `mlncData/pcg_removed_isolated_nodes/` |
| EukLoc | ~3,980 | ~116,532 | ~4 | `mlncData/EukaryoteGo/` |
| HumLoc | ~3,236 | ~13,236 | ~11 | `mlncData/HumanGo/` |

### 2. 数据文件结构

每个数据集目录应包含:

```
mlncData/dataset_name/
├── features.csv          # 节点特征 (num_nodes × feature_dim)
├── labels.csv            # 多标签矩阵 (num_nodes × num_labels)
├── edge_list.csv         # 边列表或相邻矩阵
├── split.pt              # 训练/验证/测试划分 (PyTorch)
└── [可选] 其他相关文件
```

### 3. 数据文件格式

**features.csv** (特征文件)
```
1.2,0.5,3.4,...  # 第一个节点的特征向量
0.8,1.2,2.1,...  # 第二个节点的特征向量
...
```

**labels.csv** (标签文件)
```
1,0,1,0  # 第一个节点的标签(多标签二进制编码)
0,1,1,0  # 第二个节点的标签
...
```

**split.pt** (数据划分)
```python
# 使用 torch.save() 创建
{
    'train_mask': torch.tensor([True, False, ...]),  # 训练集掩码
    'val_mask': torch.tensor([False, True, ...]),    # 验证集掩码
    'test_mask': torch.tensor([False, False, ...])   # 测试集掩码
}
```

### 4. 生成临时特征文件（仅用于测试）

如果特征文件缺失，可以生成随机特征作为临时替代:

```bash
python3.12 << 'EOF'
import numpy as np
import os

# 读取标签文件确定节点数
labels = np.loadtxt('mlncData/dblp/labels.txt', delimiter=',')
num_nodes = labels.shape[0]

# 生成随机特征 (通常维度100-300)
features = np.random.randn(num_nodes, 100).astype(np.float32)
np.savetxt('mlncData/dblp/features.txt', features, delimiter=',', fmt='%.6f')
print(f"生成特征文件: {num_nodes}x100")
EOF
```

---

## 模型训练

### 1. 基本训练命令

```bash
python3.12 main_lip.py \
  --device cpu \
  --dataset dblp \
  --model_type gcn \
  --train_ratio 0.6 \
  --test_ratio 0.2 \
  --learnCoef auto \
  --epoch 200 \
  --run 3
```

### 2. 命令行参数详解

#### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--device` | str | `cuda:0` | 计算设备: `cpu` 或 `cuda:0`, `cuda:1` 等 |
| `--dataset` | str | `EukLoc` | 数据集名称: `dblp`, `blogcatalog`, `pcg`, `EukLoc`, `HumLoc` |
| `--model_type` | str | `gcn` | 模型类型: `gcn`, `gat`, `appnp`, `homo`, `hetero` |

#### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epoch` | int | `200` | 最大训练轮数 |
| `--run` | int | `3` | 独立训练运行次数(用于统计) |
| `--patience` | int | `10` | 早停耐心值(轮数) |
| `--lr` | float | `0.01` | 学习率 |
| `--wd` | float | `0` | 权重衰减(L2正则化) |
| `--dropout` | float | `0.2` | Dropout比例 |

#### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_ratio` | float | `0.6` | 训练集比例 |
| `--test_ratio` | float | `0.2` | 测试集比例(验证集为剩余) |
| `--lbls` | list | `[0,1,2,3]` | 使用的标签索引 |

#### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--hid_dim` | int | `64` | 隐层维度 |
| `--num_layers` | int | `2` | GNN层数 |
| `--order` | int | `2` | Beta Wavelet中的阶数(仅BWGNN) |
| `--homo` | int | `1` | 1=同构图, 0=异构图 |
| `--activation` | str | `relu` | 激活函数: `relu`, `leaky_relu` |

#### 系数学习参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--learnCoef` | str | `our` | 系数学习方式: `auto`(自动学习), `grad`(梯度相似度), `our`(改进方法), `none`(不学习), `cooc`(共现矩阵) |

### 3. 常见训练场景

#### 场景 A: 快速测试（5个epoch）
```bash
python3.12 main_lip.py --device cpu --dataset dblp --model_type gcn \
  --epoch 5 --run 1
```
**耗时**: ~2秒 | **用途**: 验证环境和数据

#### 场景 B: 完整训练（推荐）
```bash
python3.12 main_lip.py --device cpu --dataset dblp --model_type gcn \
  --epoch 200 --run 3
```
**耗时**: ~40分钟 | **用途**: 获得最优性能和统计结果

#### 场景 C: GPU加速训练
```bash
python3.12 main_lip.py --device cuda:0 --dataset dblp --model_type gcn \
  --epoch 200 --run 3
```
**耗时**: ~5分钟 | **用途**: 大规模数据集或高性能需求

**注意**: 需要DGL编译CUDA支持。如果不可用，使用CPU。

#### 场景 D: 调整学习率
```bash
python3.12 main_lip.py --device cpu --dataset dblp --model_type gcn \
  --lr 0.05 --wd 1e-4 --dropout 0.3 --epoch 200
```
**用途**: 超参数调优

#### 场景 E: 不同的模型对比
```bash
# GCN
python3.12 main_lip.py --device cpu --dataset dblp --model_type gcn

# GAT (图注意力网络)
python3.12 main_lip.py --device cpu --dataset dblp --model_type gat

# APPNP (近似个性化PageRank)
python3.12 main_lip.py --device cpu --dataset dblp --model_type appnp

# BWGNN (Beta Wavelet GNN)
python3.12 main_lip.py --device cpu --dataset dblp --model_type homo
```

#### 场景 F: 不同系数学习方式对比
```bash
# 自动系数学习
python3.12 main_lip.py --device cpu --dataset dblp --learnCoef auto

# 梯度相似度
python3.12 main_lip.py --device cpu --dataset dblp --learnCoef grad

# 改进方法(推荐)
python3.12 main_lip.py --device cpu --dataset dblp --learnCoef our

# 不学习系数(等权)
python3.12 main_lip.py --device cpu --dataset dblp --learnCoef none
```

### 4. 输出解释

训练过程输出示例:
```
Epoch=50, loss=1.18, test_auc=0.685, test_ap=0.466,
best_ap=0.540, val_auc=0.737, val_ap=0.540
```

| 指标 | 说明 | 范围 |
|------|------|------|
| `loss` | 当前epoch的训练损失 | 越低越好 |
| `test_auc` | 测试集AUC值 | 0-1, 越高越好 |
| `test_ap` | 测试集平均精度 | 0-1, 越高越好 |
| `best_ap` | 迄今最佳验证集AP | 0-1, 越高越好 |
| `val_auc` | 验证集AUC值 | 0-1, 越高越好 |
| `val_ap` | 验证集平均精度 | 0-1, 越高越好 |

最终输出:
```
AP-mean: 53.72, AP-std: 0.43, AUC-mean: 72.71, AUC-std: 0.13
```
- `mean`: 3次运行的平均值
- `std`: 3次运行的标准差（越小越稳定）

---

## 项目文件结构

```
LIP_MLNC/
├── main_lip.py                 # 主训练脚本
├── AllModel.py                 # 模型定义 (GCN, GAT, BWGNN等)
├── dataset.py                  # 数据加载和预处理
├── utils.py                    # 工具函数
├── mlncData/                   # 数据目录
│   ├── dblp/                   # DBLP数据集
│   │   ├── features.txt
│   │   ├── labels.txt
│   │   └── dblp.edgelist
│   ├── blogcatalog/
│   ├── pcg_removed_isolated_nodes/
│   ├── EukaryoteGo/
│   └── HumanGo/
├── PR/                         # 预计算的PageRank矩阵
│   ├── dblp_PRcooc.npy
│   ├── EukLoc_PRcooc.npy
│   └── ...
├── wheels/                     # 离线Python包
│   ├── torch-*.whl
│   ├── dgl-*.whl
│   └── ...
├── res/                        # 结果输出目录
├── models/                     # 保存的模型目录
├── emb/                        # 嵌入输出目录
├── install_final.sh            # 自动安装脚本
└── README.md
```

---

## 常见问题解决

### 问题 1: ModuleNotFoundError: No module named 'torch'

**原因**: PyTorch未安装

**解决**:
```bash
# 使用本地轮子
cd wheels
python3.12 -m pip install --no-index --find-links ./ torch

# 或在线安装
pip install torch
```

### 问题 2: DGL requires PyTorch to be installed

**原因**: DGL找不到PyTorch，通常是版本不匹配

**解决**:
```bash
# 确保PyTorch先安装
pip install torch --upgrade

# 再安装DGL
pip install dgl
```

### 问题 3: CUDA error: no kernel image for GPU

**原因**: CUDA版本不匹配，使用了GPU不支持的CUDA计算能力

**解决**:
```bash
# 使用CPU
python3.12 main_lip.py --device cpu

# 或重新安装匹配的PyTorch CUDA版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题 4: FileNotFoundError: features.txt not found

**原因**: 数据集文件缺失

**解决**:
1. 检查数据文件是否存在
2. 生成临时特征文件（见上文）
3. 确认路径是否正确

### 问题 5: Early stopping triggered

**信息**: 这不是错误，说明模型在验证集上连续10个epoch无改进，自动停止以防止过拟合

**调整**: 修改 `--patience` 参数
```bash
python3.12 main_lip.py --patience 20  # 增加早停耐心值
```

### 问题 6: RuntimeError: CUDA out of memory

**原因**: GPU内存不足

**解决**:
```bash
# 减小batch大小（在代码中修改）
# 或使用CPU
python3.12 main_lip.py --device cpu
```

---

## 高级用法

### 1. 使用不同的数据集

```bash
# EukLoc 数据集
python3.12 main_lip.py --dataset EukLoc --epoch 200

# HumLoc 数据集
python3.12 main_lip.py --dataset HumLoc --epoch 200

# 自定义数据集（需要在dataset.py中添加支持）
# 编辑 dataset.py 的 Dataset 类，在 __init__ 中添加新的数据集加载逻辑
```

### 2. 自定义模型训练

编辑 `main_lip.py` 的第234-270行:

```python
if args.model_type == "gcn":
    model = GCN(
        in_feats, h_feats, num_classes, graph,
        args.dropout,
        num_layers=args.num_layers,      # 修改层数
        activation=activation,
        num_lbls=len(args.lbls)
    ).to(args.device)
```

### 3. 添加新模型

1. 在 `AllModel.py` 中定义新模型类
2. 在 `main_lip.py` 中的参数解析中添加选项
3. 在训练逻辑中添加模型实例化代码

### 4. 调试模式

在 `main_lip.py` 顶部添加:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 打印更多详细信息
print(f"Model: {model}")
print(f"Graph: {graph}")
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
```

### 5. 保存和加载模型

```python
# 保存模型
torch.save(model.state_dict(), './models/best_model.pt')

# 加载模型
model.load_state_dict(torch.load('./models/best_model.pt'))
```

---

## 性能优化建议

### 1. 使用GPU加速

如果服务器支持GPU:
```bash
# 检查CUDA是否可用
python3.12 -c "import torch; print(torch.cuda.is_available())"

# 使用GPU
python3.12 main_lip.py --device cuda:0
```

### 2. 调整超参数

```bash
# 增加学习率（可能收敛更快）
python3.12 main_lip.py --lr 0.02

# 增加dropout防止过拟合
python3.12 main_lip.py --dropout 0.5

# 减小权重衰减
python3.12 main_lip.py --wd 0.0001
```

### 3. 多进程并行运行

```bash
# 在后台运行多个实验
nohup python3.12 main_lip.py --dataset dblp > dblp.log 2>&1 &
nohup python3.12 main_lip.py --dataset EukLoc > eukloc.log 2>&1 &
nohup python3.12 main_lip.py --dataset HumLoc > humloc.log 2>&1 &

# 查看进程
ps aux | grep main_lip.py
```

---

## 引用信息

如果在研究中使用本项目，请引用:

```bibtex
@inproceedings{lip_mlnc_iclr2025,
  title={Multi-Label Node Classification with Label Influence Propagation},
  booktitle={ICLR 2025},
  year={2025}
}
```

---

## 联系方式和支持

如有问题或建议，请:
1. 检查本文档的FAQ部分
2. 查看代码注释
3. 参考论文补充材料

---

**文档更新时间**: 2025年2月5日
**最后修改**: Claude Code
