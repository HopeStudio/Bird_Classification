# Step3 Pytorch导出ONNX

本文档说明如何使用 `scripts/export_onnx.py`，把 PyTorch 模型权重导出为 ONNX 模型文件，并解释为什么下一步网页推理需要这一步。

本步骤默认你已经完成：

- [Step1_准备环境.md](./Step1_准备环境.md)
- [Step2_模型预测测试.md](./Step2_模型预测测试.md)

本文档中提到的“项目根目录”，指的是仓库中同时包含 `scripts/`、`models/`、`db/`、`data/`、`web/`、`notes/` 的目录。

本文档中的命令示例统一使用相对路径写法，默认都在项目根目录执行。

## 步骤1：先理解什么是 ONNX

ONNX 的全称是：

```text
Open Neural Network Exchange
```

它可以理解为一种通用的模型交换格式，用来把深度学习模型保存成跨框架、跨平台更容易复用的形式。

简单来说：

- PyTorch 原始权重文件通常是 `.pth`
- ONNX 导出文件通常是 `.onnx`
- `.pth` 更偏向 PyTorch 自己使用
- `.onnx` 更适合被其他推理引擎或其他运行环境加载

## 步骤2：为什么这里需要从 PyTorch 导出成 ONNX

当前项目里：

- `scripts/predict.py` 使用的是 PyTorch 推理
- `web/` 目录下的网页使用的是浏览器端 ONNX Runtime 推理

也就是说：

- 命令行脚本可以直接加载 `models/model20240824.pth`
- 浏览器网页不能直接加载 PyTorch 的 `.pth`
- 浏览器网页需要加载 `models/model20240824.onnx`

因此，要让下一步的网页预测跑起来，就必须先把 PyTorch 模型导出成 ONNX。

网页端之所以使用 ONNX，是因为浏览器里实际调用的是前端的 ONNX Runtime：

- `web/vendor/onnxruntime/ort.min.js`
- `web/app.js`

网页打开后，会直接读取：

- `../models/model20240824.onnx`
- `../db/bird_info.csv`

所以这一步导出 ONNX，是 Step4 静态网页服务的前置条件。

## 步骤3：确认导出脚本和输入文件

本项目用于导出的脚本是：

```text
scripts/export_onnx.py
```

默认输入和输出文件是：

```text
输入权重: models/model20240824.pth
输出模型: models/model20240824.onnx
```

也就是说，脚本默认会把：

```text
models/model20240824.pth
```

导出成：

```text
models/model20240824.onnx
```

## 步骤4：先确认 bird-classify 环境和依赖

打开终端，进入项目根目录后，先激活 Step1 中创建的环境：

```bash
conda activate bird-classify
```

然后确认导出 ONNX 所需依赖已经安装：

```bash
python -c "import torch, torchvision; print(torch.__version__)"
python -c "import onnx; print(onnx.__version__)"
python -c "import onnxscript; print('onnxscript ok')"
```

如果 `onnx` 或 `onnxscript` 未安装，可补装：

```bash
pip install onnx onnxscript
```

## 步骤5：先理解 export_onnx.py 在做什么

这个脚本的核心流程可以概括为：

1. 读取 `.pth` 权重
2. 构建一个 `resnet34(num_classes=11000)` 模型
3. 把权重加载进模型
4. 构造一个假的输入张量 `1 x 3 x 224 x 224`
5. 调用 `torch.onnx.export(...)`
6. 生成 `.onnx` 文件

脚本中的关键代码逻辑是：

```python
checkpoint = torch.load(args.model_path, map_location=device)
model = models.resnet34(num_classes=11000)
model.load_state_dict(checkpoint)
dummy_input = torch.randn(1, 3, 224, 224, device=device)

torch.onnx.export(
    model,
    dummy_input,
    args.output_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_shapes=({0: torch.export.Dim("batch_size")},),
    opset_version=args.opset,
    dynamo=True,
    external_data=False,
)
```

## 步骤6：解释 export_onnx.py 里的参数

### 6.1 `--model-path`

```python
parser.add_argument("--model-path", type=Path, default=ROOT_DIR / "models/model20240824.pth")
```

含义：

- 指定输入的 PyTorch 权重文件
- 默认值是 `models/model20240824.pth`

一般情况下不需要改。

### 6.2 `--output-path`

```python
parser.add_argument("--output-path", type=Path, default=ROOT_DIR / "models/model20240824.onnx")
```

含义：

- 指定导出的 ONNX 文件路径
- 默认输出到 `models/model20240824.onnx`

下一步网页默认读取的就是这个位置，所以建议保持默认。

### 6.3 `--opset`

```python
parser.add_argument("--opset", type=int, default=18)
```

含义：

- 指定 ONNX 的 opset 版本
- 默认是 `18`

可以把它理解为 ONNX 图结构所遵循的一套算子标准版本。

通常：

- 版本太低，可能缺少新算子支持
- 版本太高，某些推理环境可能不兼容

这里默认 `18` 是一个比较常见、较新的选择。

### 6.4 `--device`

```python
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
```

含义：

- 指定导出时使用 CPU 还是 CUDA
- 默认是 `cpu`

因为导出 ONNX 只是做一次格式转换，所以通常用 CPU 就够了。

即使你传了 `--device cuda`，脚本也会先检查：

```python
torch.cuda.is_available()
```

如果当前机器没有可用 CUDA，脚本会自动回退到 CPU。

## 步骤7：解释 torch.onnx.export(...) 里的关键参数

### 7.1 `input_names=["input"]`

含义：

- 指定导出后 ONNX 模型的输入名叫 `input`

这很重要，因为网页里推理时就是按这个名字喂数据：

```javascript
const output = await state.session.run({ input });
```

如果导出时输入名不一致，网页端就无法正常推理。

### 7.2 `output_names=["logits"]`

含义：

- 指定导出后 ONNX 模型的输出名叫 `logits`

网页端读取输出时也依赖这个名字：

```javascript
const logits = output.logits?.data;
```

如果这里改名，网页代码也要一起改。

### 7.3 `dynamic_shapes=({0: torch.export.Dim("batch_size")},)`

含义：

- 把输入张量的第 0 维，也就是 batch 维度，导出成动态维度

也就是说，模型不被固定死在只能输入 1 张图片的批量大小上。

虽然当前网页每次通常只预测 1 张图片，但保留动态 batch 会更灵活。

### 7.4 `opset_version=args.opset`

含义：

- 使用命令行参数指定的 ONNX opset 版本
- 默认值是 `18`

### 7.5 `dynamo=True`

含义：

- 使用较新的 PyTorch ONNX 导出路径

这是 PyTorch 新版推荐的导出方式之一，通常兼容性和导出行为更现代。

### 7.6 `external_data=False`

含义：

- 不把权重拆成外部数据文件
- 导出为单个 `.onnx` 文件

这对网页部署很重要，因为浏览器静态页面更适合直接读取一个完整的 `.onnx` 文件，而不是同时依赖多个权重分片文件。

## 步骤8：执行 ONNX 导出

在项目根目录执行：

```bash
python scripts/export_onnx.py --device cpu
```

如果一切正常，终端应看到类似输出：

```text
Exported ONNX model to: .../models/model20240824.onnx
```

这表示导出成功。

## 步骤9：确认 ONNX 文件已经生成

导出完成后，检查输出文件是否存在：

```bash
ls models
```

重点确认：

```text
models/model20240824.onnx
```

如果该文件存在，说明网页端推理所需模型已经准备好。

## 步骤10：为什么网页端不能直接用 .pth

这是很多读者会问的一点，这里单独说明：

- `.pth` 是 PyTorch 自己的权重格式
- 浏览器里没有 PyTorch Python 运行时
- 浏览器里运行的是 JavaScript 和 WebAssembly
- 当前网页前端集成的是 ONNX Runtime Web，而不是 PyTorch

所以浏览器端的正确做法是：

1. 在 Python 环境里把 `.pth` 导出为 `.onnx`
2. 在浏览器里加载 `.onnx`
3. 用 ONNX Runtime Web 执行推理

## 步骤11：导出成功后，下一步做什么

导出成功后，就可以进入下一步静态网页测试：

- [Step4_静态网页服务.md](./Step4_静态网页服务.md)

在下一步里，网页会直接加载：

- `models/model20240824.onnx`
- `db/bird_info.csv`

然后在浏览器中完成图片上传和预测。

## 步骤12：常见问题排查

### 12.1 报错 `No module named onnx` 或 `No module named onnxscript`

说明导出依赖没装完整，可执行：

```bash
pip install onnx onnxscript
```

### 12.2 报错找不到 `.pth` 模型文件

检查默认输入文件是否存在：

```text
models/model20240824.pth
```

也可以手动指定：

```bash
python scripts/export_onnx.py --model-path models/model20240824.pth --output-path models/model20240824.onnx --device cpu
```

### 12.3 导出成功，但网页还是加载失败

重点检查：

- 生成的文件是否就在 `models/model20240824.onnx`
- 是否通过静态 HTTP 服务访问网页
- `web/app.js` 中的模型路径是否仍是 `../models/model20240824.onnx`

### 12.4 是否一定要用 GPU 导出

不需要。

本项目导出 ONNX 时，CPU 完全够用，命令也更稳妥：

```bash
python scripts/export_onnx.py --device cpu
```

## 步骤13：最小操作流程总结

如果你只想快速完成导出，可以直接执行：

```bash
conda activate bird-classify
python scripts/export_onnx.py --device cpu
```

预期结果是生成：

```text
models/model20240824.onnx
```

生成成功后，即可继续阅读：

- [Step4_静态网页服务.md](./Step4_静态网页服务.md)
