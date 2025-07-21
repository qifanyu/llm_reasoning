# 模型评估

本项目用于评估不同训练方式下的基础四则运算模型在 GSM8K 和小学数据集（primary）上的推理性能。当前支持以下三类模型的评估：

- baseline（零微调）
- non-loop finetuned（常规方式微调）
- loop finetuned（带循环结构微调）

## 📁 项目结构

```
.
├── data                         # 数据目录
│   └── primary                 # 小学四则运算数据集
│       ├── arithmetic_dataset.json         # 主数据集文件
│       ├── generate_arithmetic_dataset.py  # 数据生成脚本
│       └── README.md
├── infer_mp.py                 # 多进程推理主脚本
├── README.md                   # 项目说明文档
├── requirements.txt            # 所需Python依赖
├── scripts
│   ├── infer_loop_finetuned.sh     # 评估 loop finetuned 模型的脚本
│   └── infer_non-loop.sh           # 评估 baseline 和 non-loop 模型的脚本
├── setup.sh                    # 环境配置脚本
└── src
    └── config.py              # 推理配置文件
```

## 🚀 快速开始

### 1. 环境配置

运行以下命令以安装依赖并设置运行环境：

```bash
sh setup.sh
```

> 该脚本会自动创建 Python 虚拟环境并安装所需依赖。

### 2. 模型评估

使用以下脚本进行不同模型的推理评估：

#### 评估 baseline 与 non-loop finetuned 模型

```bash
sh scripts/infer_non-loop.sh
```

#### 评估 loop finetuned 模型

```bash
sh scripts/infer_loop_finetuned.sh
```

每个脚本运行后会输出该模型在指定数据集（GSM8K 或 primary）上的准确率等指标。

### 3. 数据说明

项目默认使用 `data/primary/arithmetic_dataset.json` 作为小学四则运算数据，数据通过下面的脚本用 `generate_arithmetic_dataset.py` 生成：

```bash
python data/primary/generate_arithmetic_dataset.py
```

## 📊 模型评估结果

| 数据集   | baseline | baseline finetuned | loop finetuned |
|----------|----------|--------------------|----------------|
| gsm8k    | 70.89%   | 77.71%             | 82.26%         |
| primary  | 47.30%   | 58.60%             | 74.30%         |