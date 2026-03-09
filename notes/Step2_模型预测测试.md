# Step2 模型预测测试

本文档说明如何使用 `scripts/predict.py` 对测试图片进行模型预测，并给出一次实际运行结果，方便读者比对。

本次测试使用的图片为：

- `data/_Z9W0960.jpg`

本步骤默认你已经完成 [Step1_准备环境.md](./Step1_准备环境.md) 中的环境准备，并且已经创建好：

```bash
bird-classify
```

如果你本机实际环境名不是 `bird-classify`，把下面命令中的环境名替换成你自己的即可。

本文档中提到的“项目根目录”，指的是仓库中同时包含 `scripts/`、`models/`、`db/`、`data/`、`web/`、`notes/` 的目录。

本文档中的命令示例统一使用相对路径写法，默认都在项目根目录执行。

## 步骤1：确认测试所需文件存在

运行预测前，先确认以下文件已经存在：

- `scripts/predict.py`
- `data/_Z9W0960.jpg`
- `models/model20240824.pth`
- `db/bird_info.csv`

可以在项目根目录执行：

```bash
ls scripts
ls data
ls models
ls db
```

重点确认这 3 个文件：

`data/_Z9W0960.jpg`

`models/model20240824.pth`

`db/bird_info.csv`

说明：

- `predict.py` 是预测脚本
- `model20240824.pth` 是模型权重
- `bird_info.csv` 用于把类别编号映射为鸟类中英文名和简介

## 步骤2：激活 Step1 中创建的 conda 环境

打开终端后，执行：

```bash
conda activate bird-classify
```

然后检查 Python 和关键依赖是否可用：

```bash
python --version
python -c "import torch, torchvision; from PIL import Image; print('env ok')"
```

如果输出了 `env ok`，说明预测脚本的基础依赖已经准备完成。

## 步骤3：先查看预测脚本的用法

建议先执行：

```bash
python scripts/predict.py --help
```

这个脚本的主要参数有：

- `--image`：要预测的图片路径，必填
- `--mode`：预测模式，可选 `normal` 或 `tta`
- `--top-k`：输出前几个候选结果，默认是 `5`

一般测试时，先用默认模式 `normal` 即可。

## 步骤4：运行一次基础预测

在项目根目录执行：

```bash
python scripts/predict.py --image data/_Z9W0960.jpg
```

这条命令的含义是：

- 使用默认预测模式 `normal`
- 对 `data/_Z9W0960.jpg` 进行分类
- 输出前 5 个候选结果

## 步骤5：本项目一次实际运行得到的预测结果

我在当前仓库中实际运行后，得到的结果如下：

```text
预测模式: normal

最可能的结果:
model_class_id: 4004
中文名: 鸡尾鹦鹉
英文名: Cockatiel
学名: Nymphicus hollandicus
置信度: 97.03%

候选结果:
1. 鸡尾鹦鹉 / Cockatiel / Nymphicus hollandicus / 97.03%
2. 凤头树燕 / Crested Treeswift / Hemiprocne coronata / 0.64%
3. 灰腰雨燕 / Grey-rumped Treeswift / Hemiprocne longipennis / 0.47%
4. 须凤头雨燕 / Moustached Treeswift / Hemiprocne mystacea / 0.46%
5. 燕尾鸢 / Swallow-tailed Kite / Elanoides forficatus / 0.17%
```

如果你的环境、模型文件和数据文件一致，预测结果应该与上面非常接近。

其中最重要的比对项是：

- `model_class_id: 4004`
- `中文名: 鸡尾鹦鹉`
- `英文名: Cockatiel`
- `学名: Nymphicus hollandicus`
- `置信度` 大约为 `97%`

说明：

- 置信度存在极小浮动是正常现象
- 只要 Top1 结果一致，通常就说明模型加载和推理流程是正常的

## 步骤6：可选，测试 `top-k` 参数

如果你想查看更多候选结果，可以手动指定 `top-k`：

```bash
python scripts/predict.py --image data/_Z9W0960.jpg --top-k 10
```

这样会输出前 10 个候选类别，方便排查模型是否存在相近类别混淆。

## 步骤7：可选，测试 `tta` 模式

该脚本还支持 `tta` 模式，也就是测试时增强预测。执行命令如下：

```bash
python scripts/predict.py --image data/_Z9W0960.jpg --mode tta
```

说明：

- `tta` 会对原图和翻转图分别预测，再综合结果
- 通常会比 `normal` 稍慢
- 如果你只是验证环境和模型是否可用，优先用 `normal`

## 步骤8：如何判断预测测试成功

满足以下几点，基本可以判断测试成功：

- 命令可以正常执行，没有报错
- 模型权重能被正确加载
- 能输出鸟类名称，而不是只有编号
- Top1 结果为 `鸡尾鹦鹉 / Cockatiel`
- 置信度明显高于其他候选结果

## 步骤9：常见问题排查

### 9.1 报错 `No module named torch`

说明当前没有进入 Step1 创建的环境，或者环境里还没安装依赖。

可重新执行：

```bash
conda activate bird-classify
python -c "import torch, torchvision; from PIL import Image; print('env ok')"
```

### 9.2 报错找不到图片

检查图片路径是否正确：

```bash
data/_Z9W0960.jpg
```

### 9.3 报错找不到模型文件

检查下面文件是否存在：

```bash
models/model20240824.pth
```

### 9.4 报错找不到 `bird_info.csv`

可以先执行：

```bash
python scripts/export_bird_info.py
```

执行成功后再重新运行预测脚本。

## 步骤10：推荐的最小测试命令

如果你只想快速确认模型能不能跑通，记住下面这两条命令就够了：

```bash
conda activate bird-classify
python scripts/predict.py --image data/_Z9W0960.jpg
```

预期核心结果：

```text
Top1: 鸡尾鹦鹉 / Cockatiel
model_class_id: 4004
confidence: 约 97%
```
