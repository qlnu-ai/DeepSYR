# DeepSYR

DeepSYR是一个统一的深度学习框架，支持多种模型库和任务类型。框架采用模块化设计，通过适配器模式抽象不同深度学习库的差异，提供统一的接口。

## 架构设计

```
Configs → Tasks → Core（接口）→ Adapters（具体实现）
            ↑                                ↓
            └───────────── Utils ────────────┘
```

### 1. Configs → Tasks
- Configs（YAML/JSON）中指明：
  - 选择的任务类型（如 text_classification）
  - 选择的后端框架（如 transformers）
  - 任务专属的超参数（学习率、批大小等）
- Tasks 层的构造函数读取 Configs，决定后续构建哪种 DataLoader/Trainer/Predictor。

### 2. Tasks → Core 接口
- 在 Task 的 _build_components() 中，调用 Core.factory 中的 build_dataloader、build_trainer、build_predictor 并传入：
  - backend（framework）
  - task（任务类型）
  - config（配置）
- Core.factory 负责把请求路由到相应的 Adapter。

### 3. Core 接口 → Adapters 实现
- Core 定义了三大基类：
  - BaseDataLoader：get_train_loader()、get_val_loader()
  - BaseTrainer：train()、validate()
  - BasePredictor：predict()
- 对应的 Adapters（每个后端各一套）继承这些基类，并实现：
  - TransformersDataLoader, YoloDataLoader, PaddleDataLoader
  - TransformersTrainer, YoloTrainer, PaddleTrainer
  - TransformersPredictor, YoloPredictor, PaddlePredictor

### 4. Utils 贯穿各层
- logging, checkpoint, metrics, distributed 等工具，被 Core 和 Adapters 共同调用，提供：
  - 统一日志格式与级别
  - 模型检查点存取
  - 分布式训练封装
  - 通用的评估指标计算

## 目录结构

```
├── configs/                  # 配置文件
│   ├── transformer_text_cls.yaml
│   ├── yolo_detect.yaml
│   └── paddle_image_cls.yaml
├── models/                   # 模型定义
│   ├── transformers/
│   ├── yolo/
│   └── paddle/
├── core/                     # 核心接口
│   ├── trainer.py
│   ├── dataloader.py
│   ├── predictor.py
│   └── factory.py
├── tasks/                    # 任务层
│   ├── base.py
│   ├── text_classification.py
│   ├── image_classification.py
│   └── object_detection.py
├── adapters/                 # 适配器层
│   ├── transformers/
│   ├── yolo/
│   └── paddle/
├── utils/                    # 工具模块
│   ├── logging.py
│   ├── metrics.py
│   ├── checkpoint.py
│   └── distributed.py
└── examples/                 # 示例脚本
    ├── run_text_cls.py
    ├── run_img_cls.py
    └── run_detect.py
```

## 支持的任务

- 文本分类（Text Classification）
- 图像分类（Image Classification）
- 目标检测（Object Detection）

## 支持的后端

- Transformers (Hugging Face)
- YOLO (Ultralytics)
- PaddlePaddle (百度飞桨)

## 使用方法

### 安装

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python examples/run_text_cls.py --config configs/transformer_text_cls.yaml --train
```

### 评估模型

```bash
python examples/run_text_cls.py --config configs/transformer_text_cls.yaml --eval --model_path ./outputs/text_cls
```

### 预测

```bash
python examples/run_text_cls.py --config configs/transformer_text_cls.yaml --predict "这是一个测试文本" --model_path ./outputs/text_cls
```

## 扩展框架

### 添加新任务

1. 在 `tasks/` 目录下创建新的任务类，继承 `BaseTask`
2. 实现必要的方法：`prepare_data`, `prepare_model`, `train`, `evaluate`, `predict`, `save`, `load`

### 添加新后端

1. 在 `adapters/` 目录下创建新的后端目录
2. 实现相应的适配器：`data_adapter.py`, `trainer_adapter.py`, `predictor_adapter.py`

### 添加新模型

1. 在 `models/` 目录下的相应后端目录中添加新模型
2. 确保模型类名为 `Model`，或在 `factory.py` 中添加相应的导入逻辑 