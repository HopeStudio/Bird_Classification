# Step1 准备环境

本文档用于准备本项目运行所需的 Python 环境，适用于以下 3 个脚本：

- `scripts/predict.py`
- `scripts/export_bird_info.py`
- `scripts/export_onnx.py`

本文档中提到的“项目根目录”，指的是仓库中同时包含以下目录的那一层目录：

- `scripts/`
- `models/`
- `db/`
- `data/`
- `web/`
- `notes/`

数据库的原始文件直接在仓库的 Release 页面下载 `bird_reference.sqlite`，并放到 `db/` 目录中。

本文档中的命令示例统一使用相对路径写法，默认都在项目根目录执行。

## 步骤1：确认这 3 个脚本需要哪些依赖

根据脚本中的 `import` 判断：

### 1.1 Python 标准库

以下模块属于 Python 标准库，不需要单独安装：

- `argparse`
- `csv`
- `sqlite3`
- `pathlib`

### 1.2 需要额外安装的第三方库

这几个库需要安装：

- `torch`
- `torchvision`
- `pillow`

### 1.3 仅在导出 ONNX 时建议额外安装的库

`scripts/export_onnx.py` 使用了 `torch.onnx.export(...)`。为了更稳定地导出 ONNX，建议额外安装：

- `onnx`
- `onnxscript`

如果你暂时只运行图片预测 `scripts/predict.py`，可以先不安装这两个库。

## 步骤2：确认 Python 版本

这 3 个脚本本身没有写死 Python 版本要求，但结合 PyTorch 和 torchvision 的常见兼容性，建议使用：

```bash
Python 3.10
```

原因：

- 兼容性通常最好
- 安装 `torch` / `torchvision` 时踩坑更少
- 适合 CPU 版和可选 GPU 版环境

如果你已经有 Python 3.11，也大概率可以使用，但本项目更推荐先用 `Python 3.10`。

## 步骤3：安装 VS Code

为了后续更方便地查看代码、编辑文件、运行命令和阅读项目文档，建议先安装 VS Code。

VS Code 的作用：

- 方便浏览项目目录结构
- 方便打开和编辑 Python、Markdown、HTML、CSS、JavaScript 文件
- 可以直接在编辑器里打开终端
- 比较适合初学者学习和整理项目

安装完成后，建议至少确认以下几点：

- 可以正常打开 VS Code
- 可以使用 VS Code 打开本项目文件夹
- 可以在 VS Code 里打开终端

如果你后续需要运行 Python 脚本，也建议在 VS Code 中安装 Python 扩展，不过这不是必须条件。

## 步骤4：安装 Miniforge

因为这里使用 conda，推荐安装 Miniforge。

Miniforge 的作用：

- 自带 conda
- 更适合创建独立环境
- 安装科学计算相关包更方便

安装完成后，打开终端，先验证 conda 是否可用：

```bash
conda --version
```

如果能看到类似下面的输出，就说明 conda 已经安装成功：

```bash
conda 24.x.x
```

## 步骤5：创建 conda 环境

推荐创建一个单独环境，例如命名为 `bird-classify`：

```bash
conda create -n bird-classify python=3.10 -y
```

创建完成后激活环境：

```bash
conda activate bird-classify
```

## 步骤6：安装运行项目所需的基础依赖

### 6.1 安装 CPU 版本 PyTorch

如果你只是先把项目跑起来，CPU 版就够用了，建议优先使用：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow
```

说明：

- `predict.py` 需要 `torch`、`torchvision`、`pillow`
- `export_bird_info.py` 只用标准库，不需要额外装包
- `export_onnx.py` 至少需要 `torch`、`torchvision`

### 6.2 如果需要导出 ONNX，再补装这两个包

```bash
pip install onnx onnxscript
```

## 步骤7：可选，安装 GPU 版本 PyTorch

如果你的电脑已经正确安装 NVIDIA 驱动，并且希望使用 GPU 推理或导出，可以改为安装 GPU 版本。

例如，安装 CUDA 12.1 对应版本时，常见命令如下：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pillow
```

如果还要导出 ONNX：

```bash
pip install onnx onnxscript
```

注意：

- GPU 环境不是必须
- `predict.py` 会自动优先使用 CUDA，如果检测不到就回退到 CPU
- 如果你不确定 CUDA 版本，先用 CPU 版最稳妥

## 步骤8：验证 conda 环境是否创建成功

激活环境后，执行下面几条命令：

```bash
conda env list
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
python -c "from PIL import Image; print('Pillow OK')"
```

如果你安装了 ONNX 相关依赖，也可以继续验证：

```bash
python -c "import onnx; print(onnx.__version__)"
python -c "import onnxscript; print('onnxscript OK')"
```

判断标准：

- `conda env list` 中能看到 `bird-classify`
- `python --version` 显示的是 `3.10.x`
- `torch`、`torchvision`、`Pillow` 可以正常导入
- 如果装了 ONNX，`onnx` 和 `onnxscript` 也能正常导入

## 步骤9：验证脚本运行环境是否正常

### 9.1 验证预测脚本依赖

```bash
python scripts/predict.py --help
```

如果环境没问题，应该能正常显示命令行参数说明。

### 9.2 验证导出鸟类信息脚本

```bash
python scripts/export_bird_info.py
```

如果运行成功，通常会在 `db/` 目录下生成：

```bash
db/bird_info.csv
```

### 9.3 验证 ONNX 导出脚本

先确认已经安装：

- `torch`
- `torchvision`
- `onnx`
- `onnxscript`

然后执行：

```bash
python scripts/export_onnx.py --device cpu
```

如果成功，通常会在 `models/` 目录下生成 `.onnx` 文件。

## 步骤10：推荐的一次性安装方案

如果你想直接一次装好 CPU 版环境，可以按下面顺序执行：

```bash
conda create -n bird-classify python=3.10 -y
conda activate bird-classify
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow onnx onnxscript
```

如果你只想先跑预测，不需要导出 ONNX，可以少装两个包：

```bash
conda create -n bird-classify python=3.10 -y
conda activate bird-classify
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow
```

## 步骤11：本项目依赖总结

### 必装

- `python=3.10`
- `torch`
- `torchvision`
- `pillow`

### 按需安装

- `onnx`
- `onnxscript`

### 不需要单独安装

- `argparse`
- `csv`
- `sqlite3`
- `pathlib`
