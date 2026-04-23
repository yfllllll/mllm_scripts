好的，我将把您提供的配置说明整合到 README 中，使其更加详细和友好。以下是更新后的 `README.md` 完整内容：

```markdown
# 目标检测评估实验脚本使用说明

## 功能概述

本工具用于批量评估视觉语言模型在目标检测/图像分类任务上的性能。支持：
- 多种模型（通过 API 调用）
- 多个数据集（每个数据集可自定义检测类别）
- 多种提示词模板（中英文、检测模式/VQA模式）
- 生成带边界框的可视化图片，并自动计算查准率、查全率、F1、准确率等指标
- 输出 CSV 汇总表和自包含的 HTML 报告（可直接复制到腾讯文档）

## 环境准备

### 安装依赖

```bash
pip install openai Pillow
```

### 文件结构

```
project/
├── detector_evaluator.py   # 核心评估模块
├── run_experiments.py      # 批量实验脚本
└── README.md               # 本说明文档
```

### 启动 API 服务

确保 vLLM 或其他兼容 OpenAI API 的服务已在运行，并记录其地址（例如 `http://localhost:9007/v1`）。

## 配置实验

编辑 `run_experiments.py` 中的配置区域，根据实际需求修改以下三个核心配置项。

### 1. 模型列表 `MODELS`

```python
MODELS = [
    {"name": "QwenVl-7B", "api_base": "http://localhost:9007/v1"},
    # 可添加更多模型，每个模型可指定独立的 API 地址
]
```

- `name`：模型名称，会显示在报告和输出目录中。
- `api_base`：该模型对应的 API 服务地址。

### 2. 任务配置 `TASKS`

```python
TASKS = {
    "烟火数据_v1": {
        "path": "/path/to/dataset1",                # 数据集路径，下需有 positive/ 和 negative/ 子文件夹
        "categories": ["夜间火焰", "夜间烟雾", "夜间烟囱烟雾"]
    },
    "烟火数据_v2": {
        "path": "/path/to/dataset2",
        "categories": ["火焰", "烟雾"]                # 不同数据集可使用不同类别
    },
}
```

- 字典的键为任务名称（自定义）。
- `path`：数据集根目录，必须包含 `positive/` 和 `negative/` 两个子文件夹。
- `categories`：该数据集需要检测的目标类别列表，支持中文或英文。

### 3. 提示词模板 `PROMPTS`

```python
PROMPTS = {
    "detection_cn": "检测图像中的{categories}。以 JSON 格式报告边界框坐标。",
    "vqa_cn": "请仔细观察图像，判断图中是否存在{categories}。只需回答“是”或“否”。",
    # 可添加更多模板，{categories} 会被自动替换
}
```

- 键名以 `detection` 开头的使用**检测模式**（模型返回边界框 JSON）。
- 键名以 `vqa` 开头的使用**问答模式**（模型返回“是/否”文本）。
- 模板中的 `{categories}` 会被自动替换为当前任务的类别列表（以顿号连接）。

### 4. 输出根目录 `DEFAULT_OUTPUT_ROOT`

```python
DEFAULT_OUTPUT_ROOT = "/mnt/disk/liangqh/experiments"
```

- 所有实验结果将保存在该路径下，每个实验批次会创建一个带时间戳的子目录（如 `exp_20250422_143000`）。

## 运行实验

### 首次运行（自动创建带时间戳的新目录）
```bash
python run_experiments.py
```

### 追加到已有实验目录
```bash
python run_experiments.py --exp_root /path/to/existing/exp_20250422_143000
```
脚本会自动读取目录下的 `all_experiments_summary.csv`，跳过已存在的实验ID，仅运行新实验，然后更新汇总表和 HTML 报告。

### 强制覆盖已存在的实验
```bash
python run_experiments.py --exp_root /path/to/existing/exp_20250422_143000 --force
```

## 输出结果说明

每个实验目录（如 `exp_20250422_143000/`）下包含：

- **各实验子目录**（如 `QwenVl-7B_烟火数据_v1_detection_cn/`）：
  - `TP/`, `TN/`, `FP/`, `FN/` 四个文件夹，存放分类后的图像（TP和FP为带框图像，TN和FN为原图）
  - `details.csv`：每张图像的详细检测结果和响应文本
  - `summary.csv`：该实验的统计指标

- **根目录汇总文件**：
  - `all_experiments_summary.csv`：所有实验的指标汇总表
  - `report.html`：图文并茂的 HTML 报告，包含汇总表、提示词、示例图片

### HTML 报告使用技巧
- 用浏览器打开 `report.html`，全选复制后可直接粘贴到腾讯文档中，保留表格和图片。
- 报告中的图片使用相对路径引用，因此**整个实验目录**需保持完整，移动时请整体移动。

## 单独运行单个评估

也可以直接调用 `detector_evaluator.py` 进行单次评估：

```bash
python detector_evaluator.py \
    --input_dir /path/to/dataset \
    --output_dir /path/to/output \
    --api_base http://localhost:9007/v1 \
    --model QwenVl \
    --mode detection \
    --delay 1.0
```

## 常见问题

### Q: 数据集目录结构要求？
A: 每个数据集目录下必须有 `positive/` 和 `negative/` 两个子文件夹，图像格式支持 `.jpg/.jpeg/.png/.bmp/.tiff/.webp`。

### Q: 如何修改检测类别？
A: 在 `TASKS` 配置中为每个数据集指定 `categories` 列表即可。

### Q: 报告中的图片无法显示？
A: 确保 HTML 文件与各实验子目录的相对位置未改变。图片路径格式为 `实验ID/分类/图片名`。

### Q: 想调整图像缩放大小？
A: 修改 `detector_evaluator.py` 中 `resize_image_max_side` 的 `max_size` 参数（默认1280）。

## 联系与维护

如有问题，请检查 API 服务是否正常，或查看运行日志中的错误信息。
```

在脚本 `run_experiments.py` 中，**追加**和**覆盖**的工作逻辑基于实验唯一标识符（`exp_id`）和已有汇总文件的记录。下面详细解释判断流程：

---

### 实验唯一标识符（`exp_id`）

`exp_id` 的生成规则：
```python
exp_id = f"{model_name}_{task_name}_{prompt_key}"
```
例如：`QwenVl-7B_烟火数据_v1_detection_cn`

这个 ID 在整个实验目录中是**唯一**的，对应一个子目录和一组结果。

---

### 启动时读取已有实验记录

函数 `load_existing_summaries(exp_root)` 会尝试读取 `exp_root/all_experiments_summary.csv`：
- 若文件存在，提取其中所有 `experiment_id`，存入 `existing_ids` 集合。
- 若文件不存在，则 `existing_ids` 为空。

---

### 遍历每个实验配置时的判断逻辑

对于每个待运行的实验（由 `MODELS` × `TASKS` × `PROMPTS` 组合产生），执行以下检查：

```python
if exp_id in existing_ids and not args.force:
    print(f"实验 {exp_id} 已存在，跳过（使用 --force 可强制覆盖）")
    continue
```

#### 情况 1：实验 ID **不在**已有记录中
- 直接创建输出子目录并运行评估。
- 完成后将统计结果加入 `new_summaries`，并将 `exp_id` 加入 `existing_ids`（用于后续去重）。

#### 情况 2：实验 ID **已在**已有记录中，且**未使用** `--force`
- 跳过该实验，不运行任何计算。

#### 情况 3：实验 ID **已在**已有记录中，且**使用了** `--force`
- 检查输出子目录是否存在，若存在则**删除整个目录**（`shutil.rmtree`）。
- 重新创建目录并运行评估。
- 结果覆盖原有记录。

---

### 汇总文件的更新

无论运行了多少个新实验，最后都会执行合并操作：

1. 将 `existing_records`（从 CSV 读出的旧记录）和 `new_summaries`（本次新运行的记录）合并为 `all_summaries`。
2. **完全重写** `all_experiments_summary.csv`。
3. **完全重新生成** `report.html`，包含所有历史实验和新实验的数据。

因此，追加的本质是**保留已有子目录和原记录，仅新增未运行过的实验**，而覆盖是**强制删除特定子目录，重新计算并更新总表**。

---

### 示意图

```
实验目录/
├── all_experiments_summary.csv   ← 已有记录：exp_A, exp_B
├── exp_A/                         ← 已存在
├── exp_B/                         ← 已存在
└── report.html

运行 python run_experiments.py --exp_root 实验目录
  → 遇到 exp_A → 跳过
  → 遇到 exp_C → 运行，新增 exp_C/，记录加入总表
  → 重写 all_experiments_summary.csv（包含 exp_A, exp_B, exp_C）
  → 重写 report.html

运行 python run_experiments.py --exp_root 实验目录 --force
  → 遇到 exp_A → 删除 exp_A/，重新运行，覆盖记录
```

这种设计既支持**增量追加实验**，也允许**单独重新运行某个失败的实验**。