const MODEL_URL = "../models/model20240824.onnx";
const CSV_URL = "../db/bird_info.csv";
const INPUT_SIZE = 224;
const RESIZE_SIZE = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const $ = (id) => document.getElementById(id);
const els = {
  imageInput: $("image-input"),
  modeSelect: $("mode-select"),
  topkSelect: $("topk-select"),
  predictButton: $("predict-button"),
  modelStatus: $("model-status"),
  csvStatus: $("csv-status"),
  runStatus: $("run-status"),
  previewBox: $("preview-box"),
  resultsRoot: $("results-root"),
  resultsMeta: $("results-meta")
};

const state = {
  session: null,
  birdInfoMap: new Map(),
  modelClassCount: 0,
  selectedFile: null,
  modelReady: false,
  csvReady: false
};

function setStatus(element, text, error = false) {
  element.textContent = text;
  element.classList.toggle("error", error);
}

function updatePredictButton() {
  const ready = state.modelReady && state.csvReady;
  els.predictButton.disabled = !ready;
  els.predictButton.textContent = ready ? "开始预测" : "加载资源中...";
}

function parseCsv(text) {
  const rows = [];
  const source = text.replace(/^\uFEFF/, "");
  let row = [];
  let cell = "";
  let inQuotes = false;

  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    const next = source[i + 1];

    if (char === "\"") {
      if (inQuotes && next === "\"") {
        cell += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      row.push(cell);
      cell = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some(Boolean)) rows.push(row);
      row = [];
      cell = "";
      continue;
    }

    cell += char;
  }

  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }

  if (rows.length === 0) return [];

  const headers = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(
    headers.map((header, index) => [header, values[index] ?? ""])
  ));
}

async function loadBirdInfo() {
  setStatus(els.csvStatus, "标签状态：正在加载 bird_info.csv ...");
  const response = await fetch(CSV_URL);
  if (!response.ok) {
    throw new Error(`bird_info.csv 加载失败：${response.status} ${response.statusText}`);
  }

  const map = new Map();
  for (const row of parseCsv(await response.text())) {
    const classId = Number.parseInt((row.model_class_id || "").trim(), 10);
    if (Number.isNaN(classId)) continue;
    map.set(classId, {
      model_class_id: String(classId),
      chinese_simplified: row.chinese_simplified || "未知",
      english_name: row.english_name || "Unknown",
      scientific_name: row.scientific_name || "",
      short_description_zh: row.short_description_zh || ""
    });
  }

  state.birdInfoMap = map;
  state.modelClassCount = map.size ? Math.max(...map.keys()) + 1 : 0;
  state.csvReady = true;
  setStatus(els.csvStatus, `标签状态：已加载 ${map.size} 条类别信息`);
  updatePredictButton();
}

async function loadModel() {
  setStatus(els.modelStatus, "模型状态：正在加载 ONNX 模型，首次可能需要几十秒...");
  if (!window.ort) {
    throw new Error("onnxruntime-web 未加载成功，请检查网络或 CDN。");
  }

  ort.env.wasm.numThreads = 1;
  state.session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });

  state.modelReady = true;
  setStatus(els.modelStatus, "模型状态：模型已加载，可以开始预测");
  updatePredictButton();
}

function renderPreview(file) {
  const url = URL.createObjectURL(file);
  els.previewBox.innerHTML = `<img src="${url}" alt="上传图片预览">`;
  els.previewBox.querySelector("img").addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
}

function showEmptyResult(message) {
  els.resultsRoot.innerHTML = `<div class="empty">${message}</div>`;
}

function getBirdInfo(classId) {
  return state.birdInfoMap.get(classId) || {
    model_class_id: String(classId),
    chinese_simplified: "未知",
    english_name: "Unknown",
    scientific_name: "",
    short_description_zh: ""
  };
}

function createPreprocessCanvas(bitmap, flip = false) {
  const scale = RESIZE_SIZE / Math.min(bitmap.width, bitmap.height);
  const resizedWidth = Math.round(bitmap.width * scale);
  const resizedHeight = Math.round(bitmap.height * scale);
  const cropX = Math.max(0, Math.floor((resizedWidth - INPUT_SIZE) / 2));
  const cropY = Math.max(0, Math.floor((resizedHeight - INPUT_SIZE) / 2));

  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = resizedWidth;
  resizedCanvas.height = resizedHeight;

  const resizedContext = resizedCanvas.getContext("2d", { willReadFrequently: true });
  resizedContext.imageSmoothingEnabled = true;
  resizedContext.imageSmoothingQuality = "high";
  resizedContext.save();
  if (flip) {
    resizedContext.translate(resizedWidth, 0);
    resizedContext.scale(-1, 1);
  }
  resizedContext.drawImage(bitmap, 0, 0, resizedWidth, resizedHeight);
  resizedContext.restore();

  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(resizedCanvas, cropX, cropY, INPUT_SIZE, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
  return canvas;
}

function canvasToTensor(canvas) {
  const { data } = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const tensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const channelSize = INPUT_SIZE * INPUT_SIZE;

  for (let i = 0; i < channelSize; i += 1) {
    const offset = i * 4;
    const r = data[offset] / 255;
    const g = data[offset + 1] / 255;
    const b = data[offset + 2] / 255;
    tensor[i] = (r - MEAN[0]) / STD[0];
    tensor[channelSize + i] = (g - MEAN[1]) / STD[1];
    tensor[channelSize * 2 + i] = (b - MEAN[2]) / STD[2];
  }

  return tensor;
}

async function runModel(canvas) {
  const input = new ort.Tensor("float32", canvasToTensor(canvas), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const output = await state.session.run({ input });
  const logits = output.logits?.data;
  if (!logits) throw new Error("模型输出中未找到 logits。");
  return Array.from(logits).slice(0, state.modelClassCount);
}

function averageLogits(logitsA, logitsB) {
  return logitsA.map((value, index) => (value + logitsB[index]) / 2);
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - maxLogit));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

function getTopK(probabilities, topK) {
  return probabilities
    .map((confidence, classId) => ({ classId, confidence }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, topK);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function renderResults(results, mode) {
  if (results.length === 0) {
    showEmptyResult("没有可显示的结果。");
    return;
  }

  els.resultsMeta.textContent = `模式：${mode} · 共显示 ${results.length} 条结果`;
  els.resultsRoot.innerHTML = `
    <div class="result-list">
      ${results.map((item, index) => `
        <article class="result-card ${index === 0 ? "top1" : ""}">
          <h3>${index + 1}. ${escapeHtml(item.chineseName)}</h3>
          <p class="latin">${escapeHtml(item.englishName)}${item.scientificName ? ` · ${escapeHtml(item.scientificName)}` : ""}</p>
          <div class="confidence">class ${item.classId} · ${(item.confidence * 100).toFixed(2)}%</div>
          <p class="description">${escapeHtml(item.description || "暂无简介。")}</p>
        </article>
      `).join("")}
    </div>
  `;
}

async function predict() {
  if (!state.selectedFile) throw new Error("请先选择一张图片。");
  if (!state.selectedFile.type.startsWith("image/")) throw new Error("上传文件不是图片，请重新选择。");
  if (!state.session || !state.modelReady || !state.csvReady) throw new Error("模型或标签信息尚未加载完成。");

  const bitmap = await createImageBitmap(state.selectedFile);
  const topK = Number.parseInt(els.topkSelect.value, 10);
  const mode = els.modeSelect.value;

  try {
    let logits = await runModel(createPreprocessCanvas(bitmap, false));
    if (mode === "tta") {
      logits = averageLogits(logits, await runModel(createPreprocessCanvas(bitmap, true)));
    }

    renderResults(
      getTopK(softmax(logits), topK).map(({ classId, confidence }) => {
        const info = getBirdInfo(classId);
        return {
          classId,
          confidence,
          chineseName: info.chinese_simplified,
          englishName: info.english_name,
          scientificName: info.scientific_name,
          description: info.short_description_zh
        };
      }),
      mode
    );
  } finally {
    bitmap.close();
  }
}

els.imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files || [];
  state.selectedFile = file || null;
  els.resultsMeta.textContent = state.selectedFile ? "图片已选择，等待预测" : "等待图片";
  if (state.selectedFile) {
    renderPreview(state.selectedFile);
  } else {
    els.previewBox.innerHTML = '<div class="preview-placeholder">上传图片后，这里会显示预览。</div>';
  }
});

els.predictButton.addEventListener("click", async () => {
  setStatus(els.runStatus, "推理状态：正在执行推理...");
  els.predictButton.disabled = true;
  try {
    await predict();
    setStatus(els.runStatus, "推理状态：预测完成");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(els.runStatus, `推理状态：${message}`, true);
    showEmptyResult(escapeHtml(message));
  } finally {
    updatePredictButton();
  }
});

async function initialize() {
  try {
    await Promise.all([loadModel(), loadBirdInfo()]);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(els.runStatus, `推理状态：初始化失败：${message}`, true);
    if (!state.modelReady) setStatus(els.modelStatus, `模型状态：${message}`, true);
    if (!state.csvReady) setStatus(els.csvStatus, `标签状态：${message}`, true);
    showEmptyResult("初始化失败。请确认当前运行在静态服务器环境中，并且模型与 CSV 文件路径可访问。");
  }
}

initialize();
