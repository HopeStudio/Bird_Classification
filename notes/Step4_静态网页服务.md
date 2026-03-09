# Step4 静态网页服务

本文档说明如何使用 `web/` 目录下的静态网页，在浏览器中完成鸟类图片预测。

本步骤默认你已经完成：

- [Step1_准备环境.md](./Step1_准备环境.md)
- [Step2_模型预测测试.md](./Step2_模型预测测试.md)
- [Step3_Pytorch导出ONNX.md](./Step3_Pytorch导出ONNX.md)

本文档中提到的“项目根目录”，指的是仓库中同时包含 `scripts/`、`models/`、`db/`、`data/`、`web/`、`notes/` 的目录。

本文档中的命令示例统一使用相对路径写法，默认都在项目根目录执行。

并且已经具备以下文件：

- `web/index.html`
- `web/app.js`
- `web/styles.css`
- `models/model20240824.onnx`
- `db/bird_info.csv`

说明：

- 网页会直接在浏览器端加载 `ONNX` 模型并完成推理
- 不依赖 Python Web 后端接口
- 但必须通过静态 HTTP 服务访问，不能直接双击打开 `index.html`

## 步骤1：确认网页相关文件已经存在

在项目根目录下，先确认以下目录和文件存在：

```bash
ls web
ls models
ls db
```

重点确认：

`web/index.html`

`web/app.js`

`web/styles.css`

`models/model20240824.onnx`

`db/bird_info.csv`

其中：

- `web/` 目录存放网页文件
- `model20240824.onnx` 是浏览器端推理所需模型
- `bird_info.csv` 用于显示鸟类名称和简介

## 步骤2：激活 Step1 中创建的 conda 环境

打开终端，先进入项目根目录：

```bash
cd <项目根目录>
```

然后激活环境：

```bash
conda activate bird-classify
```

虽然启动静态文件服务本身不依赖 PyTorch，但为了与前面文档保持一致，仍然建议在 `bird-classify` 环境中完成这一步。

## 步骤3：在项目根目录启动 HTTP Server

在 `src` 根目录执行：

```bash
python -m http.server 8000
```

如果启动成功，终端通常会显示类似信息：

```text
Serving HTTP on 0.0.0.0 port 8000 ...
```

注意：

- 这条命令需要在项目根目录执行
- 不要切换到 `web/` 目录再启动
- 因为页面内部读取的是：
  - `../models/model20240824.onnx`
  - `../db/bird_info.csv`

如果你在错误目录启动服务，页面能打开，但模型或标签文件可能加载失败。

## 步骤4：在浏览器中打开网页

启动静态服务后，在浏览器访问：

```text
http://localhost:8000/web/
```

不要直接打开：

```text
web/index.html
```

原因是：

- 页面内部使用 `fetch` 读取模型和 CSV 文件
- 直接双击打开本地 HTML 时，浏览器通常会拦截这类本地文件访问
- 使用 HTTP Server 才能正常加载资源

## 步骤5：观察页面初始化状态

网页打开后，页面会自动加载：

- `../models/model20240824.onnx`
- `../db/bird_info.csv`

左侧区域通常会看到几项状态信息，例如：

- 模型状态
- 标签状态
- 推理状态

正常情况下：

- 模型状态会从“等待加载”变为“模型已加载，可以开始预测”
- 标签状态会显示已加载的类别信息数量
- 预测按钮会从不可点击变成可点击

如果页面一直显示加载失败，优先检查：

- 是否是通过 `http://localhost:8000/web/` 打开的
- `models/model20240824.onnx` 是否存在
- `db/bird_info.csv` 是否存在
- 启动服务时是否位于项目根目录

## 步骤6：上传测试图片

页面左侧有“上传图片”区域，点击后选择测试图片：

```text
data/_Z9W0960.jpg
```

上传完成后：

- 页面会显示图片预览
- 结果区域会进入“等待预测”的状态

## 步骤7：选择预测模式

页面里有“预测模式”下拉框，可选：

- `normal`
- `tta`

建议首次测试先选：

```text
normal
```

说明：

- `normal`：只对原图做一次预测
- `tta`：对原图和水平翻转图分别预测，再做平均
- `tta` 一般会比 `normal` 更慢一点

如果你只是验证网页流程是否跑通，优先选择 `normal`。

## 步骤8：选择结果数量

页面里还有“结果数量”下拉框，常见可选值为：

- `1`
- `3`
- `5`
- `10`

建议首次测试保持默认：

```text
5
```

这样可以同时看到 Top1 和若干候选结果，便于对比。

## 步骤9：点击“预测”按钮查看结果

在图片上传完成、模型和标签加载完成后，点击页面中的预测按钮。

点击后页面会：

- 更新“推理状态”
- 在右侧结果区显示预测结果
- 展示鸟类中文名、英文名、学名、类别编号和置信度

如果流程正常，本次测试图片 `data/_Z9W0960.jpg` 的核心结果应接近：

```text
Top1: 鸡尾鹦鹉
English: Cockatiel
Scientific Name: Nymphicus hollandicus
class_id: 4004
confidence: 约 97%
```

只要 Top1 结果一致，通常就说明：

- 网页成功加载了 ONNX 模型
- 图片预处理流程正常
- 浏览器端推理正常
- `bird_info.csv` 标签映射正常

## 步骤10：对照 Step2 的命令行结果

为了验证网页端结果是否可信，可以和 Step2 的命令行预测结果进行对比。

Step2 中命令行预测的核心结果是：

```text
model_class_id: 4004
中文名: 鸡尾鹦鹉
英文名: Cockatiel
学名: Nymphicus hollandicus
置信度: 97.03%
```

网页端与命令行端在预处理逻辑上是对齐的，因此结果应当非常接近。

如果你在网页端也看到了相同的 Top1 类别，说明前后两种推理方式基本一致。

## 步骤11：结束静态服务

测试完成后，回到启动 HTTP Server 的终端，按：

```bash
Ctrl + C
```

即可停止服务。

## 步骤12：常见问题排查

### 12.1 页面能打开，但按钮一直不可点击

通常表示模型或 CSV 还没有加载成功。请检查：

- `models/model20240824.onnx` 是否存在
- `db/bird_info.csv` 是否存在
- 是否通过 `http://localhost:8000/web/` 打开网页

### 12.2 页面提示模型加载失败

重点检查：

- 是否已经先完成 ONNX 导出
- `models/model20240824.onnx` 路径是否正确
- 启动静态服务时是否位于项目根目录

### 12.3 页面提示标签加载失败

重点检查：

- `db/bird_info.csv` 是否存在
- 如果不存在，可先执行：

```bash
python scripts/export_bird_info.py
```

### 12.4 直接双击 `index.html` 后页面异常

这是常见情况，原因是浏览器直接打开本地 HTML 时，页面中的 `fetch` 无法正常访问模型和 CSV 文件。

正确方式一定是：

```bash
python -m http.server 8000
```

然后访问：

```text
http://localhost:8000/web/
```

## 步骤13：最小操作流程总结

如果你只想快速跑通网页预测，最小步骤如下：

```bash
cd <项目根目录>
conda activate bird-classify
python -m http.server 8000
```

然后在浏览器中：

1. 打开 `http://localhost:8000/web/`
2. 上传 `data/_Z9W0960.jpg`
3. 选择 `normal`
4. 保持 `top-k = 5`
5. 点击预测
6. 查看结果是否为 `鸡尾鹦鹉 / Cockatiel`
