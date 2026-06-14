import * as THREE from "three";
import "./styles.css";

const PANEL = {
  width: 6.2,
  depth: 7.2,
  leftX: -3.45,
  rightX: 3.45
};
const MAX_TRAIN_SAMPLES = 640;
const REPLAY_POINTS = 520;
const FEATURE_SIZE = 9;
const EVAL_POINTS = 256;
const MATCH_DISTANCE = 0.1;
const STROKE_SAMPLE_STEP = 0.035;
const BASE_TRAIN_EPOCHS = 320;
const EXTRA_TRAIN_EPOCHS = 220;
const TRAIN_CHUNK_SIZE = 20;
const TRAIN_LOOP_YIELD_CHUNKS = 6;
const LEARNING_RATE = 0.024;
const TINY_DRAW_HIDDEN = 12;
const TINY_DISTILL_EPOCHS = 180;
const TINY_DISTILL_LR = 0.028;
const TINY_DISTILL_BATCH = 24;
const TINY_DISTILL_STRIDE = 18;
const COMPACT_MODEL_VERSION = 1;
const DEFAULT_MODEL_HIDDEN_SIZE = 96;
const SWARM_EVO_GENERATIONS = 4;
const SWARM_EVO_POPULATION = 96;
const SWARM_EVO_PARENT_POOL = 24;
const SWARM_EVO_ELITE_COUNT = 10;
const SWARM_EVO_MUTATION = 0.32;
const SWARM_EVO_EVAL_POINTS = 72;
const SWARM_AGENT_COUNT = 1200;
const SWARM_POINTS_PER_AGENT = 96;
const SWARM_SEGMENTS_PER_AGENT = SWARM_POINTS_PER_AGENT - 1;
const SWARM_VERTEX_COUNT = SWARM_AGENT_COUNT * SWARM_SEGMENTS_PER_AGENT * 2;
const SWARM_AGENT_COLORS = buildAgentColors(SWARM_AGENT_COUNT);
const SWARM_VIS_BATCH_SIZE = 8;
const RUN_STORAGE_KEY = "mirror-neuron-training:run:v1";
const RUN_HISTORY_KEY = "mirror-neuron-training:history:v1";
const RUN_HISTORY_LIMIT = 16;
const AUTO_REPLAY_DELAY_MS = 120;

const canvas = document.querySelector("#stage");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14120f);
scene.fog = new THREE.Fog(0x14120f, 10, 24);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const drawPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const hitPoint = new THREE.Vector3();

scene.add(new THREE.HemisphereLight(0xfff2df, 0x162125, 2.2));

const keyLight = new THREE.DirectionalLight(0xffefd9, 3.8);
keyLight.position.set(-4, 8, 4.5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
scene.add(keyLight);

const stage = new THREE.Group();
scene.add(stage);

const materials = {
  base: new THREE.MeshStandardMaterial({ color: 0x1b1814, roughness: 0.86 }),
  leftPanel: new THREE.MeshStandardMaterial({ color: 0x263c39, roughness: 0.78 }),
  rightPanel: new THREE.MeshStandardMaterial({ color: 0x332c43, roughness: 0.78 }),
  divider: new THREE.MeshStandardMaterial({ color: 0xe7d7bf, roughness: 0.7 }),
  pen: new THREE.LineBasicMaterial({ color: 0xffbf69, transparent: true, opacity: 0.96 }),
  ai: new THREE.LineBasicMaterial({ color: 0x5de0b5, transparent: true, opacity: 0.94 }),
  target: new THREE.LineBasicMaterial({ color: 0xffbf69, transparent: true, opacity: 0.46 }),
  faintPen: new THREE.LineBasicMaterial({ color: 0xffbf69, transparent: true, opacity: 0.28 }),
  faintAi: new THREE.LineBasicMaterial({ color: 0x5de0b5, transparent: true, opacity: 0.28 }),
  swarm: new THREE.LineBasicMaterial({
    transparent: true,
    opacity: 0.32,
    vertexColors: true,
    depthWrite: false,
    depthTest: false,
    blending: THREE.NormalBlending,
    toneMapped: false
  }),
  node: new THREE.MeshStandardMaterial({
    color: 0x8fe8ff,
    roughness: 0.35,
    emissive: 0x12333b,
    emissiveIntensity: 0.65
  })
};

const state = {
  isDrawing: false,
  isTraining: false,
  isReplaying: false,
  isRestoredRun: false,
  rawStroke: [],
  trainStroke: [],
  aiStroke: [],
  bestAiStroke: [],
  bestAiScore: null,
  bestAiEvaluation: null,
  bestAiSource: null,
  replayCursor: 0,
  replayStartedAt: 0,
  loss: null,
  epochs: 0,
  evaluation: null,
  model: null,
  compactModel: null,
  trainingSamples: [],
  trainingRunId: 0,
  swarmGeneration: 0,
  swarmDrawProgress: 0,
  swarmPopulationSize: SWARM_EVO_POPULATION,
  swarmGenerationHistory: [],
  swarmBestScore: null,
  swarmBestAgentPolicy: null,
  swarmGenerationPlan: []
};
let renderFrame = 0;

const runStatus = document.querySelector("#runStatus");
const scoreValue = document.querySelector("#scoreValue");
const lossValue = document.querySelector("#lossValue");
const pointValue = document.querySelector("#pointValue");
const epochValue = document.querySelector("#epochValue");
const agentValue = document.querySelector("#agentValue");
const coverageValue = document.querySelector("#coverageValue");
const precisionValue = document.querySelector("#precisionValue");
const curveValue = document.querySelector("#curveValue");
const lengthValue = document.querySelector("#lengthValue");
const orderValue = document.querySelector("#orderValue");
const generationValue = document.querySelector("#generationValue");
const keepLearning = document.querySelector("#keepLearning");
const repeatBest = document.querySelector("#repeatBest");
const trainMoreButton = document.querySelector("#trainMoreButton");
const resetButton = document.querySelector("#resetButton");

function buildStage() {
  const base = new THREE.Mesh(new THREE.BoxGeometry(14.4, 0.18, 8.8), materials.base);
  base.position.y = -0.13;
  base.receiveShadow = true;
  stage.add(base);

  addPanel(PANEL.leftX, materials.leftPanel);
  addPanel(PANEL.rightX, materials.rightPanel);

  const divider = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.08, PANEL.depth + 0.25), materials.divider);
  divider.position.y = 0.035;
  divider.castShadow = true;
  stage.add(divider);

  addGrid(PANEL.leftX);
  addGrid(PANEL.rightX);
  addLabels();
}

function addPanel(centerX, material) {
  const panel = new THREE.Mesh(new THREE.BoxGeometry(PANEL.width, 0.045, PANEL.depth), material);
  panel.position.set(centerX, -0.02, 0);
  panel.receiveShadow = true;
  stage.add(panel);
}

function addGrid(centerX) {
  const fineMat = new THREE.LineBasicMaterial({ color: 0xf2e3ce, transparent: true, opacity: 0.075 });
  const majorMat = new THREE.LineBasicMaterial({ color: 0xf2e3ce, transparent: true, opacity: 0.18 });
  const divisions = 64;
  for (let i = 0; i <= divisions; i += 1) {
    const x = centerX - PANEL.width / 2 + (PANEL.width * i) / divisions;
    const z = -PANEL.depth / 2 + (PANEL.depth * i) / divisions;
    const mat = i % 8 === 0 ? majorMat : fineMat;
    stage.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x, 0.035, -PANEL.depth / 2),
          new THREE.Vector3(x, 0.035, PANEL.depth / 2)
        ]),
        mat
      )
    );
    stage.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(centerX - PANEL.width / 2, 0.036, z),
          new THREE.Vector3(centerX + PANEL.width / 2, 0.036, z)
        ]),
        mat
      )
    );
  }
}

function addLabels() {
  const labelCanvas = document.createElement("canvas");
  labelCanvas.width = 1024;
  labelCanvas.height = 128;
  const ctx = labelCanvas.getContext("2d");
  ctx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
  ctx.font = "700 54px Inter, system-ui, sans-serif";
  ctx.fillStyle = "rgba(246, 240, 232, 0.84)";
  ctx.textAlign = "center";
  ctx.fillText("human drawing", 256, 82);
  ctx.fillText("AI copy", 768, 82);

  const texture = new THREE.CanvasTexture(labelCanvas);
  const label = new THREE.Mesh(
    new THREE.PlaneGeometry(12.6, 1.55),
    new THREE.MeshBasicMaterial({ map: texture, transparent: true })
  );
  label.rotation.x = -Math.PI / 2;
  label.position.set(0, 0.06, -4.08);
  stage.add(label);
}

function createLine(material) {
  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-999, 0, -999),
    new THREE.Vector3(-999, 0, -999)
  ]);
  const line = new THREE.Line(geometry, material);
  stage.add(line);
  return line;
}

function updateLine(line, points) {
  const safe = points.length > 1 ? points : points.length === 1 ? [points[0], points[0]] : [];
  if (!safe.length) {
    line.visible = false;
    return;
  }
  line.visible = true;
  line.geometry.dispose();
  line.geometry = new THREE.BufferGeometry().setFromPoints(safe);
}

function clampToZero(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function safeParseJSON(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function serializeStroke(points) {
  return points.map((point) => ({
    x: clampToZero(point.x),
    y: clampToZero(point.y),
    z: clampToZero(point.z),
    time: clampToZero(point.time, 0),
  }));
}

function deserializeStroke(points) {
  return points.map((point) => {
    const vector = new THREE.Vector3(point?.x ?? 0, point?.y ?? 0, point?.z ?? 0);
    vector.time = clampToZero(point?.time, 0);
    return vector;
  });
}

function serializeModel(model) {
  return {
    inputSize: model.inputSize,
    outputSize: model.outputSize,
    hiddenSize: model.hiddenSize,
    w1: model.w1,
    b1: model.b1,
    w2: model.w2,
    b2: model.b2,
  };
}

function serializeCompactModel(model) {
  if (!model || typeof model !== "object") return null;
  const inputSize = model.inputSize ?? FEATURE_SIZE;
  const hiddenSize = model.hiddenSize ?? 1;
  const outputSize = model.outputSize ?? 2;

  const w1 = flattenNumberMatrix(model.w1, inputSize, hiddenSize);
  const b1 = flattenNumberArray(model.b1, hiddenSize);
  const w2 = flattenNumberMatrix(model.w2, hiddenSize, outputSize);
  const b2 = flattenNumberArray(model.b2, outputSize);

  const totalLength = w1.length + b1.length + w2.length + b2.length;
  const flat = new Float32Array(totalLength);
  let cursor = 0;
  flat.set(w1, cursor);
  cursor += w1.length;
  flat.set(b1, cursor);
  cursor += b1.length;
  flat.set(w2, cursor);
  cursor += w2.length;
  flat.set(b2, cursor);

  let maxAbs = 0;
  for (let i = 0; i < flat.length; i += 1) {
    const absValue = Math.abs(flat[i]);
    if (Number.isFinite(absValue) && absValue > maxAbs) {
      maxAbs = absValue;
    }
  }
  const scale = maxAbs > 0 ? maxAbs / 127 : 1;

  const quantized = new Int8Array(flat.length);
  for (let i = 0; i < flat.length; i += 1) {
    quantized[i] = Math.max(-128, Math.min(127, Math.round(flat[i] / scale)));
  }

  return {
    v: COMPACT_MODEL_VERSION,
    i: inputSize,
    h: hiddenSize,
    o: outputSize,
    s: clampModelPayload(scale, 1),
    b: bytesToBase64(quantized)
  };
}

function deserializeModel(payload) {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const model = new StrokeNet(payload.hiddenSize ?? DEFAULT_MODEL_HIDDEN_SIZE);
  model.inputSize = payload.inputSize ?? FEATURE_SIZE;
  model.outputSize = payload.outputSize ?? 2;
  model.hiddenSize = payload.hiddenSize ?? DEFAULT_MODEL_HIDDEN_SIZE;
  if (Array.isArray(payload.w1) && payload.w1.length > 0) {
    model.w1 = payload.w1;
  }
  if (Array.isArray(payload.b1) && payload.b1.length > 0) {
    model.b1 = payload.b1;
  }
  if (Array.isArray(payload.w2) && payload.w2.length > 0) {
    model.w2 = payload.w2;
  }
  if (Array.isArray(payload.b2) && payload.b2.length > 0) {
    model.b2 = payload.b2;
  }
  return model;
}

function deserializeCompactModel(payload) {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  if (payload.v !== COMPACT_MODEL_VERSION && payload.v !== "1") {
    return null;
  }

  const inputSize = Number(payload.i) || FEATURE_SIZE;
  const hiddenSize = Number(payload.h) || 1;
  const outputSize = Number(payload.o) || 2;
  const scale = clampModelPayload(payload.s, 1);
  if (!Number.isFinite(scale) || scale <= 0) return null;

  const bytes = base64ToBytes(payload.b);
  if (!bytes || bytes.length === 0) return null;
  const quantized = new Int8Array(bytes);

  const expectedLength = inputSize * hiddenSize + hiddenSize + hiddenSize * outputSize + outputSize;
  if (quantized.length < expectedLength) return null;

  const flat = new Float32Array(expectedLength);
  for (let i = 0; i < expectedLength; i += 1) {
    flat[i] = quantized[i] * scale;
  }

  let cursor = 0;
  const w1Flat = flat.slice(cursor, cursor + inputSize * hiddenSize);
  cursor += inputSize * hiddenSize;
  const b1Flat = flat.slice(cursor, cursor + hiddenSize);
  cursor += hiddenSize;
  const w2Flat = flat.slice(cursor, cursor + hiddenSize * outputSize);
  cursor += hiddenSize * outputSize;
  const b2Flat = flat.slice(cursor, cursor + outputSize);

  const model = new StrokeNet(hiddenSize);
  model.inputSize = inputSize;
  model.outputSize = outputSize;
  model.hiddenSize = hiddenSize;
  model.w1 = unflattenToMatrix(w1Flat, inputSize, hiddenSize);
  model.b1 = Array.from(b1Flat);
  model.w2 = unflattenToMatrix(w2Flat, hiddenSize, outputSize);
  model.b2 = Array.from(b2Flat);

  return model;
}

function flattenNumberMatrix(source, rowCount, columnCount) {
  const flat = new Float32Array(rowCount * columnCount);
  let index = 0;
  for (let row = 0; row < rowCount; row += 1) {
    const sourceRow = Array.isArray(source?.[row]) ? source[row] : [];
    for (let column = 0; column < columnCount; column += 1) {
      const value = sourceRow[column];
      flat[index] = Number.isFinite(value) ? value : 0;
      index += 1;
    }
  }
  return flat;
}

function flattenNumberArray(source, length) {
  const flat = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const value = source?.[index];
    flat[index] = Number.isFinite(value) ? value : 0;
  }
  return flat;
}

function unflattenToMatrix(flat, rowCount, columnCount) {
  const matrix = [];
  let index = 0;
  for (let row = 0; row < rowCount; row += 1) {
    const rowValues = new Array(columnCount);
    for (let column = 0; column < columnCount; column += 1) {
      rowValues[column] = Number.isFinite(flat[index]) ? flat[index] : 0;
      index += 1;
    }
    matrix.push(rowValues);
  }
  return matrix;
}

function bytesToBase64(bytes) {
  const byteArray = Array.from(bytes);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < byteArray.length; i += chunk) {
    binary += String.fromCharCode(...byteArray.slice(i, i + chunk).map((value) => value & 255));
  }
  return btoa(binary);
}

function base64ToBytes(base64) {
  if (!base64 || typeof base64 !== "string") return null;
  let binary;
  try {
    binary = atob(base64);
  } catch {
    return null;
  }
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function compactModelMetadata(model) {
  const compact = serializeCompactModel(model);
  if (!compact || !compact.b) {
    return null;
  }
  return {
    v: compact.v,
    i: compact.i,
    h: compact.h,
    o: compact.o,
    s: compact.s,
    bytes: compact.b.length
  };
}

function serializeSwarmPolicy(policy) {
  if (!policy || typeof policy !== "object") return null;
  return {
    biasX: clampModelPayload(policy.biasX, 0),
    biasZ: clampModelPayload(policy.biasZ, 0),
    track: clampModelPayload(policy.track, 0),
    xAmp: clampModelPayload(policy.xAmp, 0),
    zAmp: clampModelPayload(policy.zAmp, 0),
    xFreq: clampModelPayload(policy.xFreq, 1),
    zFreq: clampModelPayload(policy.zFreq, 1),
    xPhase: clampModelPayload(policy.xPhase, 0),
    zPhase: clampModelPayload(policy.zPhase, 0),
    jitter: clampModelPayload(policy.jitter, 0),
  };
}

function deserializeSwarmPolicy(payload) {
  if (!payload || typeof payload !== "object") return null;
  return {
    biasX: clampModelPayload(payload.biasX, 0),
    biasZ: clampModelPayload(payload.biasZ, 0),
    track: clampModelPayload(payload.track, 0),
    xAmp: clampModelPayload(payload.xAmp, 0),
    zAmp: clampModelPayload(payload.zAmp, 0),
    xFreq: clampModelPayload(payload.xFreq, 1),
    zFreq: clampModelPayload(payload.zFreq, 1),
    xPhase: clampModelPayload(payload.xPhase, 0),
    zPhase: clampModelPayload(payload.zPhase, 0),
    jitter: clampModelPayload(payload.jitter, 0),
  };
}

function clampModelPayload(value, fallback = null) {
  if (!Number.isFinite(value)) return fallback;
  return value;
}

function readRunHistory() {
  const stored = localStorage.getItem(RUN_HISTORY_KEY);
  const parsed = safeParseJSON(stored ?? "");
  if (!Array.isArray(parsed)) return [];
  return parsed.filter((entry) => entry && entry.timestamp);
}

function saveRunHistory(record) {
  const history = readRunHistory();
  const nextHistory = [record, ...history].filter((entry, index, list) =>
    list.findIndex((item) => item.timestamp === entry.timestamp) === index
  ).slice(0, RUN_HISTORY_LIMIT);
  localStorage.setItem(RUN_HISTORY_KEY, JSON.stringify(nextHistory));
}

function persistRunSnapshot(context = {}) {
  if (typeof localStorage === "undefined" || !state.rawStroke.length) {
    return;
  }

  const record = {
    version: 1,
    timestamp: Date.now(),
    keepLearning: keepLearning.checked,
    rawStroke: serializeStroke(state.rawStroke),
    loss: state.loss,
    epochs: state.epochs,
    evaluation: state.evaluation,
    model: state.model ? serializeModel(state.model) : null,
    bestAiScore: state.bestAiScore,
    swarmGeneration: state.swarmGeneration,
    swarmDrawProgress: state.swarmDrawProgress,
    swarmPopulationSize: state.swarmPopulationSize,
    swarmBestScore: state.swarmBestScore,
    swarmBestAgentPolicy: serializeSwarmPolicy(state.swarmBestAgentPolicy),
    swarmGenerationHistory: state.swarmGenerationHistory,
    trainingRunId: state.trainingRunId,
    status: state.isTraining ? "training" : "complete",
    ...context,
  };

  localStorage.setItem(RUN_STORAGE_KEY, JSON.stringify(record));
  saveRunHistory(record);
}

function clearPersistedRun() {
  localStorage.removeItem(RUN_STORAGE_KEY);
  localStorage.removeItem(RUN_HISTORY_KEY);
}

function hydrateFromRunRecord(record) {
  if (!record || typeof record !== "object") {
    return false;
  }

  state.rawStroke = deserializeStroke(Array.isArray(record.rawStroke) ? record.rawStroke : []);
  state.trainingSamples = Array.isArray(record.trainingSamples) && record.trainingSamples.length
    ? record.trainingSamples
    : buildTrainingSamples(state.rawStroke);
  state.aiStroke = [];
  state.loss = clampModelPayload(record.loss, null);
  state.epochs = Number.isFinite(record.epochs) ? record.epochs : 0;
  state.evaluation = record.evaluation ?? null;
  state.swarmGeneration = Number.isFinite(record.swarmGeneration) ? record.swarmGeneration : 0;
  state.swarmDrawProgress = Number.isFinite(record.swarmDrawProgress) ? record.swarmDrawProgress : 0;
  state.swarmPopulationSize = Number.isFinite(record.swarmPopulationSize) ? record.swarmPopulationSize : SWARM_EVO_POPULATION;
  state.swarmBestScore = clampModelPayload(record.swarmBestScore, null);
  state.swarmBestAgentPolicy = deserializeSwarmPolicy(record.swarmBestAgentPolicy);
  state.swarmGenerationHistory = Array.isArray(record.swarmGenerationHistory) ? record.swarmGenerationHistory : [];
  state.trainingRunId = Number.isFinite(record.trainingRunId) ? record.trainingRunId : 0;
  state.bestAiScore = clampModelPayload(record.bestAiScore, null);
  state.bestAiStroke = [];
  state.bestAiEvaluation = null;
  state.bestAiSource = null;
  state.compactModel = null;
  state.model = record.model
    ? (deserializeModel(record.model) || deserializeCompactModel(record.model))
    : null;
  state.isTraining = false;
  state.isRestoredRun = true;

  if (record.keepLearning != null) {
    keepLearning.checked = Boolean(record.keepLearning);
  }

  if (state.rawStroke.length > 0) {
    updateLine(humanLine, state.rawStroke);
    pointValue.textContent = state.rawStroke.length.toLocaleString();
  }

  if (state.trainingSamples.length > 0) {
    updateLine(targetLine, trainingSamplesToWorld(state.trainingSamples));
    if (state.swarmBestAgentPolicy) {
      const targetPath = state.trainingSamples.map((sample) => ({ x: sample.target[0], z: sample.target[1] }));
      const bestPath = buildCandidatePathFromPolicy(targetPath, state.swarmBestAgentPolicy);
      updateSwarmFromPolicyPlan([{ policy: state.swarmBestAgentPolicy, path: bestPath, score: state.swarmBestScore, index: 0 }], 1);
    } else {
      updateSwarmFromSamples(state.trainingSamples, 0.94);
    }
    updateSwarmDrawProgress(state.swarmDrawProgress);
  }

  if (state.model) {
    const baseEvaluation = generateAiStroke(state.model);
    if (baseEvaluation?.accepted || Number.isFinite(state.bestAiScore)) {
      state.bestAiStroke = state.aiStroke.slice();
      state.bestAiEvaluation = baseEvaluation.evaluation;
      state.bestAiSource = "restored";
      updateLine(aiGhost, state.bestAiStroke);
      state.evaluation = state.bestAiEvaluation;
      state.bestAiScore = Number.isFinite(baseEvaluation.score) ? baseEvaluation.score : state.bestAiScore;
      state.aiStroke = state.bestAiStroke.slice();
      updateLine(aiLine, []);
    }
  }
  if (state.bestAiStroke.length > 0) {
    scheduleAutoReplay();
  }

  updateUi("Loaded persistent training run.");
  return true;
}

function loadPersistedRun() {
  const stored = localStorage.getItem(RUN_STORAGE_KEY);
  if (!stored) return false;
  const record = safeParseJSON(stored);
  return hydrateFromRunRecord(record);
}

function makeCursor(color) {
  const cursor = new THREE.Mesh(
    new THREE.SphereGeometry(0.075, 20, 12),
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.3,
      emissive: color,
      emissiveIntensity: 0.28
    })
  );
  cursor.position.set(-999, 0, -999);
  cursor.castShadow = true;
  stage.add(cursor);
  return cursor;
}

function createSwarmMesh() {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(SWARM_VERTEX_COUNT * 3);
  const colors = new Float32Array(SWARM_VERTEX_COUNT * 3);
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage));
  geometry.setDrawRange(0, 0);

  const lines = new THREE.LineSegments(geometry, materials.swarm);
  lines.frustumCulled = false;
  lines.renderOrder = 20;
  stage.add(lines);
  return lines;
}

function createNeuralCloud() {
  const cloud = new THREE.Group();
  for (let i = 0; i < 34; i += 1) {
    const node = new THREE.Mesh(new THREE.SphereGeometry(0.035 + Math.random() * 0.035, 12, 8), materials.node);
    node.position.set(
      (Math.random() - 0.5) * 1.2,
      0.45 + Math.random() * 1.5,
      (Math.random() - 0.5) * 5.4
    );
    node.userData.phase = Math.random() * Math.PI * 2;
    cloud.add(node);
  }
  stage.add(cloud);
  return cloud;
}

buildStage();
const humanLine = createLine(materials.pen);
const aiLine = createLine(materials.ai);
const targetLine = createLine(materials.target);
targetLine.renderOrder = 120;
targetLine.material.depthTest = false;
targetLine.material.depthWrite = false;
const humanGhost = createLine(materials.faintPen);
const aiGhost = createLine(materials.faintAi);
const swarmMesh = createSwarmMesh();
const humanCursor = makeCursor(0xffbf69);
const aiCursor = makeCursor(0x5de0b5);
const neuralCloud = createNeuralCloud();

class StrokeNet {
  constructor(hiddenSize = 96) {
    this.inputSize = FEATURE_SIZE;
    this.hiddenSize = hiddenSize;
    this.outputSize = 2;
    this.w1 = randomMatrix(this.inputSize, hiddenSize, 0.24);
    this.b1 = new Array(hiddenSize).fill(0);
    this.w2 = randomMatrix(hiddenSize, this.outputSize, 0.2);
    this.b2 = new Array(this.outputSize).fill(0);
  }

  clone() {
    const copy = new StrokeNet(this.hiddenSize);
    copy.inputSize = this.inputSize;
    copy.outputSize = this.outputSize;
    copy.w1 = this.w1.map((row) => row.slice());
    copy.b1 = this.b1.slice();
    copy.w2 = this.w2.map((row) => row.slice());
    copy.b2 = this.b2.slice();
    return copy;
  }

  forward(input) {
    const hidden = new Array(this.hiddenSize);
    for (let h = 0; h < this.hiddenSize; h += 1) {
      let sum = this.b1[h];
      for (let i = 0; i < this.inputSize; i += 1) sum += input[i] * this.w1[i][h];
      hidden[h] = Math.tanh(sum);
    }

    const output = new Array(this.outputSize);
    for (let o = 0; o < this.outputSize; o += 1) {
      let sum = this.b2[o];
      for (let h = 0; h < this.hiddenSize; h += 1) sum += hidden[h] * this.w2[h][o];
      output[o] = Math.tanh(sum);
    }
    return { hidden, output };
  }

  train(samples, epochs, learningRate, onProgress) {
    let finalLoss = 0;
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      let totalLoss = 0;
      for (const sample of samples) {
        const { hidden, output } = this.forward(sample.input);
        const dOut = [output[0] - sample.target[0], output[1] - sample.target[1]];
        totalLoss += dOut[0] * dOut[0] + dOut[1] * dOut[1];

        dOut[0] *= 2 * (1 - output[0] * output[0]);
        dOut[1] *= 2 * (1 - output[1] * output[1]);

        const dHidden = new Array(this.hiddenSize).fill(0);
        for (let h = 0; h < this.hiddenSize; h += 1) {
          dHidden[h] += this.w2[h][0] * dOut[0];
          dHidden[h] += this.w2[h][1] * dOut[1];
        }

        const dPreHidden = dHidden.map((grad, h) => grad * (1 - hidden[h] * hidden[h]));
        for (let h = 0; h < this.hiddenSize; h += 1) {
          this.w2[h][0] -= learningRate * hidden[h] * dOut[0];
          this.w2[h][1] -= learningRate * hidden[h] * dOut[1];
          this.b1[h] -= learningRate * dPreHidden[h];
          for (let i = 0; i < this.inputSize; i += 1) {
            this.w1[i][h] -= learningRate * sample.input[i] * dPreHidden[h];
          }
        }
        this.b2[0] -= learningRate * dOut[0];
        this.b2[1] -= learningRate * dOut[1];
      }
      finalLoss = totalLoss / samples.length;
      if ((epoch % 20 === 0 || epoch === epochs - 1)) {
        onProgress?.(epoch + 1, finalLoss);
      }
    }
    return finalLoss;
  }
}

function randomMatrix(rows, cols, scale) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
  );
}

function buildDistillSamples(sourceSamples, teacherModel, options = {}) {
  if (!Array.isArray(sourceSamples) || !teacherModel || !sourceSamples.length) {
    return [];
  }

  const stride = Math.max(1, Number(options.stride) || TINY_DISTILL_STRIDE);
  const limit = Math.max(1, Math.min(sourceSamples.length, Number(options.maxSamples) || sourceSamples.length));
  const out = [];

  for (let index = 0; index < limit; index += stride) {
    const sample = sourceSamples[index];
    if (!sample || !Array.isArray(sample.input) || sample.input.length < FEATURE_SIZE) continue;
    const output = teacherModel.forward(sample.input).output;
    out.push({
      input: sample.input.slice(),
      target: [clampModelPayload(output[0], 0), clampModelPayload(output[1], 0)]
    });
  }

  if (out.length < 2) {
    const fallback = [sourceSamples[0], sourceSamples[sourceSamples.length - 1]];
    for (const sample of fallback) {
      if (!sample || !Array.isArray(sample.input)) continue;
      const output = teacherModel.forward(sample.input).output;
      out.push({
        input: sample.input.slice(),
        target: [clampModelPayload(output[0], 0), clampModelPayload(output[1], 0)]
      });
    }
  }

  return out;
}

function trainDistillModel(teacherModel, sourceSamples, options = {}) {
  if (!teacherModel) return null;
  const distillSamples = buildDistillSamples(
    sourceSamples,
    teacherModel,
    {
      stride: options.stride ?? TINY_DISTILL_STRIDE,
      maxSamples: options.maxSamples ?? sourceSamples?.length
    }
  );
  if (distillSamples.length < 2) return null;

  const model = new StrokeNet(TINY_DRAW_HIDDEN);
  const epochs = Math.max(1, Number(TINY_DISTILL_EPOCHS) || 1);
  const learningRate = Number.isFinite(options.learningRate) ? options.learningRate : TINY_DISTILL_LR;
  const batch = Math.max(1, Number(TINY_DISTILL_BATCH) || 1);

  let totalLoss = null;
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    totalLoss = model.train(distillSamples, batch, learningRate, null);
  }

  return {
    model,
    loss: totalLoss,
    sampleCount: distillSamples.length,
    epochCount: epochs
  };
}

function buildAgentColors(count) {
  const colors = new Float32Array(count * 3);
  const color = new THREE.Color();
  const goldenRatio = 0.61803398875;
  let hue = 0.09;

  for (let i = 0; i < count; i += 1) {
    hue = (hue + goldenRatio) % 1;
    const saturation = 0.92 + hashUnit(i, 21) * 0.08;
    const lightness = 0.7 + hashUnit(i, 22) * 0.22;
    color.setHSL(hue, saturation, lightness);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  return colors;
}

function features(t) {
  return [
    1,
    t,
    t * t,
    Math.sin(Math.PI * 2 * t),
    Math.cos(Math.PI * 2 * t),
    Math.sin(Math.PI * 4 * t),
    Math.cos(Math.PI * 4 * t),
    Math.sin(Math.PI * 8 * t),
    Math.cos(Math.PI * 8 * t)
  ];
}

function leftWorldToNorm(point) {
  return {
    x: THREE.MathUtils.clamp((point.x - PANEL.leftX) / (PANEL.width / 2), -1, 1),
    z: THREE.MathUtils.clamp(point.z / (PANEL.depth / 2), -1, 1),
    time: point.time
  };
}

function worldToNorm(point, centerX) {
  return {
    x: THREE.MathUtils.clamp((point.x - centerX) / (PANEL.width / 2), -1, 1),
    z: THREE.MathUtils.clamp(point.z / (PANEL.depth / 2), -1, 1)
  };
}

function normToWorld(point, centerX) {
  return new THREE.Vector3(
    centerX + point.x * (PANEL.width / 2),
    0.09,
    point.z * (PANEL.depth / 2)
  );
}

function isInsideLeftPanel(point) {
  return (
    point.x >= PANEL.leftX - PANEL.width / 2 &&
    point.x <= PANEL.leftX + PANEL.width / 2 &&
    point.z >= -PANEL.depth / 2 &&
    point.z <= PANEL.depth / 2
  );
}

function eventToLeftPoint(event) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  raycaster.setFromCamera(pointer, camera);
  if (!raycaster.ray.intersectPlane(drawPlane, hitPoint)) return null;
  if (!isInsideLeftPanel(hitPoint)) return null;
  return new THREE.Vector3(hitPoint.x, 0.09, hitPoint.z);
}

function addStrokePoint(point) {
  const last = state.rawStroke[state.rawStroke.length - 1];
  if (last && last.distanceToSquared(point) < 0.00035) return;

  const now = performance.now();
  point.time = now;
  if (last) {
    const distance = last.distanceTo(point);
    const inserts = Math.max(1, Math.ceil(distance / STROKE_SAMPLE_STEP));
    for (let i = 1; i <= inserts; i += 1) {
      const t = i / inserts;
      const sampled = last.clone().lerp(point, t);
      sampled.time = THREE.MathUtils.lerp(last.time ?? now, now, t);
      state.rawStroke.push(sampled);
    }
  } else {
    state.rawStroke.push(point.clone());
  }

  updateLine(humanLine, state.rawStroke);
  humanCursor.position.copy(point);
  pointValue.textContent = state.rawStroke.length.toLocaleString();
}

function clearStroke() {
  clearPersistedRun();
  state.trainingRunId += 1;
  state.isDrawing = false;
  state.isTraining = false;
  state.isReplaying = false;
  state.rawStroke = [];
  state.trainStroke = [];
  state.aiStroke = [];
  state.replayCursor = 0;
  state.loss = null;
  state.epochs = 0;
  state.evaluation = null;
  state.model = null;
  state.compactModel = null;
  state.trainingSamples = [];
  state.bestAiStroke = [];
  state.bestAiScore = null;
  state.bestAiEvaluation = null;
  state.bestAiSource = null;
  state.swarmGeneration = 0;
  state.swarmDrawProgress = 0;
  state.swarmPopulationSize = SWARM_EVO_POPULATION;
  state.swarmBestScore = null;
  state.swarmBestAgentPolicy = null;
  state.swarmGenerationHistory = [];
  state.swarmGenerationPlan = [];
  humanCursor.position.set(-999, 0, -999);
  aiCursor.position.set(-999, 0, -999);
  updateLine(humanLine, []);
  updateLine(aiLine, []);
  updateLine(targetLine, []);
  updateLine(humanGhost, []);
  updateLine(aiGhost, []);
  swarmMesh.geometry.setDrawRange(0, 0);
  updateUi("Draw on the left grid. Release to train.");
}

function normalizeStroke(rawStroke) {
  const firstTime = rawStroke[0]?.time ?? 0;
  const lastTime = rawStroke[rawStroke.length - 1]?.time ?? firstTime + 1;
  const duration = Math.max(1, lastTime - firstTime);
  return rawStroke.map((point, index) => {
    const normalized = leftWorldToNorm(point);
    normalized.t = rawStroke.length === 1 ? 0 : (point.time - firstTime) / duration;
    normalized.indexT = rawStroke.length === 1 ? 0 : index / (rawStroke.length - 1);
    return normalized;
  });
}

function resampleStroke(stroke, count) {
  if (stroke.length <= count) return stroke;
  const sampled = [];
  for (let i = 0; i < count; i += 1) {
    const sourceIndex = Math.round((i / (count - 1)) * (stroke.length - 1));
    sampled.push(stroke[sourceIndex]);
  }
  return sampled;
}

function buildTrainingSamples(rawStroke) {
  const normalized = normalizeStroke(rawStroke);
  const sampled = resampleByArcLength(normalized, Math.min(MAX_TRAIN_SAMPLES, normalized.length));
  return sampled.map((point, index) => ({
    input: features(sampled.length === 1 ? 0 : index / (sampled.length - 1)),
    target: [point.x, point.z]
  }));
}

function trainingSamplesToWorld(samples) {
  return samples.map((sample) => {
    const point = normToWorld({ x: sample.target[0], z: sample.target[1] }, PANEL.rightX);
    point.y = 0.115;
    return point;
  });
}

function hashUnit(seed, salt) {
  const value = Math.sin((seed + 1) * 127.1 + salt * 311.7) * 43758.5453;
  return value - Math.floor(value);
}

function random01(seed, salt) {
  return hashUnit(seed, salt + 0.31);
}

function normalizePolicyValue(value, fallback, min = -Infinity, max = Infinity) {
  if (!Number.isFinite(value)) return fallback;
  return THREE.MathUtils.clamp(value, min, max);
}

function randomRange(seed, salt, min, max) {
  return min + (max - min) * random01(seed, salt);
}

function randomSigned(seed, salt, maxMagnitude) {
  return (random01(seed, salt) * 2 - 1) * maxMagnitude;
}

function createSwarmPolicy(seedValue) {
  const seed = Number.isFinite(seedValue) ? seedValue : Math.floor(Math.random() * 0x7fffffff);
  return {
    seed,
    biasX: randomSigned(seed, 11, 0.22),
    biasZ: randomSigned(seed, 12, 0.22),
    track: randomRange(seed, 13, 0.15, 0.92),
    xAmp: randomRange(seed, 14, 0.0, 0.44),
    zAmp: randomRange(seed, 15, 0.0, 0.44),
    xFreq: randomRange(seed, 16, 0.25, 4.75),
    zFreq: randomRange(seed, 17, 0.25, 4.75),
    xPhase: randomRange(seed, 18, 0, Math.PI * 2),
    zPhase: randomRange(seed, 19, 0, Math.PI * 2),
    jitter: randomRange(seed, 20, 0.0, 0.11)
  };
}

function cloneSwarmPolicy(policy) {
  return {
    seed: Number.isFinite(policy?.seed) ? policy.seed : Math.floor(Math.random() * 0x7fffffff),
    biasX: normalizePolicyValue(policy?.biasX, 0, -0.22, 0.22),
    biasZ: normalizePolicyValue(policy?.biasZ, 0, -0.22, 0.22),
    track: normalizePolicyValue(policy?.track, 0.5, 0.05, 1),
    xAmp: normalizePolicyValue(policy?.xAmp, 0, 0, 0.5),
    zAmp: normalizePolicyValue(policy?.zAmp, 0, 0, 0.5),
    xFreq: normalizePolicyValue(policy?.xFreq, 1.2, 0.05, 6),
    zFreq: normalizePolicyValue(policy?.zFreq, 1.2, 0.05, 6),
    xPhase: normalizePolicyValue(policy?.xPhase, 0, 0, Math.PI * 2),
    zPhase: normalizePolicyValue(policy?.zPhase, 0, 0, Math.PI * 2),
    jitter: normalizePolicyValue(policy?.jitter, 0, 0, 0.2)
  };
}

function mutateSwarmPolicy(policy, mutationAmount = SWARM_EVO_MUTATION) {
  const mutation = normalizePolicyValue(mutationAmount, SWARM_EVO_MUTATION, 0, 1);
  const clone = cloneSwarmPolicy(policy);
  const mag = Math.max(0.005, mutation);

  clone.seed = (clone.seed * 1664525 + 1013904223) | 0;
  clone.biasX = normalizePolicyValue(clone.biasX + randomSigned(clone.seed, 1, 0.12 * mag), 0, -0.22, 0.22);
  clone.biasZ = normalizePolicyValue(clone.biasZ + randomSigned(clone.seed, 2, 0.12 * mag), 0, -0.22, 0.22);
  clone.track = normalizePolicyValue(clone.track + randomSigned(clone.seed, 3, 0.23 * mag), 0.05, 0.99);
  clone.xAmp = normalizePolicyValue(clone.xAmp + randomSigned(clone.seed, 4, 0.2 * mag), 0, 0.5);
  clone.zAmp = normalizePolicyValue(clone.zAmp + randomSigned(clone.seed, 5, 0.2 * mag), 0, 0.5);
  clone.xFreq = normalizePolicyValue(clone.xFreq + randomSigned(clone.seed, 6, 0.9 * mag), 0.2, 6);
  clone.zFreq = normalizePolicyValue(clone.zFreq + randomSigned(clone.seed, 7, 0.9 * mag), 0.2, 6);
  clone.xPhase = normalizePolicyValue(clone.xPhase + randomSigned(clone.seed, 8, Math.PI * 0.25 * mag), 0, Math.PI * 2);
  clone.zPhase = normalizePolicyValue(clone.zPhase + randomSigned(clone.seed, 9, Math.PI * 0.25 * mag), 0, Math.PI * 2);
  clone.jitter = normalizePolicyValue(clone.jitter + randomSigned(clone.seed, 10, 0.03 * mag), 0, 0.15);
  return clone;
}

function blendSwarmPolicies(a, b) {
  const child = cloneSwarmPolicy({
    seed: Math.floor((a.seed + b.seed + Math.random() * 2e9) % 0x7fffffff),
    biasX: Math.random() < 0.5 ? a.biasX : b.biasX,
    biasZ: Math.random() < 0.5 ? a.biasZ : b.biasZ,
    track: Math.random() < 0.5 ? a.track : b.track,
    xAmp: Math.random() < 0.5 ? a.xAmp : b.xAmp,
    zAmp: Math.random() < 0.5 ? a.zAmp : b.zAmp,
    xFreq: Math.random() < 0.5 ? a.xFreq : b.xFreq,
    zFreq: Math.random() < 0.5 ? a.zFreq : b.zFreq,
    xPhase: Math.random() < 0.5 ? a.xPhase : b.xPhase,
    zPhase: Math.random() < 0.5 ? a.zPhase : b.zPhase,
    jitter: Math.random() < 0.5 ? a.jitter : b.jitter
  });

  return mutateSwarmPolicy(child, SWARM_EVO_MUTATION * (0.55 + Math.random() * 0.45));
}

function buildLegacySwarmPolicy(agentIndex, learning) {
  const base = createSwarmPolicy(agentIndex + learning * 1000 + 1);
  base.track = normalizePolicyValue(0.72 + learning * 0.25 + randomSigned(agentIndex, 24, 0.08), 0.82, 0.55, 1);
  base.xAmp = normalizePolicyValue(base.xAmp * (1 - learning) * 1.8, 0, 0.12);
  base.zAmp = normalizePolicyValue(base.zAmp * (1 - learning) * 1.8, 0, 0.12);
  base.jitter = normalizePolicyValue(base.jitter * (1 - learning) * 2.2, 0, 0.12);
  return base;
}

function buildCandidatePathFromPolicy(basePath, policy) {
  if (!basePath || basePath.length < 2) return [];
  const safePath = resampleByArcLength(basePath, SWARM_POINTS_PER_AGENT);
  const points = [];
  const track = normalizePolicyValue(policy?.track, 0.5, 0.05, 1);
  const ampX = normalizePolicyValue(policy?.xAmp, 0, 0, 0.5);
  const ampZ = normalizePolicyValue(policy?.zAmp, 0, 0, 0.5);
  const freqX = normalizePolicyValue(policy?.xFreq, 1.2, 0.15, 6);
  const freqZ = normalizePolicyValue(policy?.zFreq, 1.2, 0.15, 6);
  const phaseX = normalizePolicyValue(policy?.xPhase, 0, 0, Math.PI * 2);
  const phaseZ = normalizePolicyValue(policy?.zPhase, 0, 0, Math.PI * 2);
  const jitter = normalizePolicyValue(policy?.jitter, 0, 0, 0.2);
  const biasX = normalizePolicyValue(policy?.biasX, 0, -0.22, 0.22);
  const biasZ = normalizePolicyValue(policy?.biasZ, 0, -0.22, 0.22);

  for (let index = 0; index < safePath.length; index += 1) {
    const point = safePath[index];
    const t = index / (safePath.length - 1);
    const sinX = Math.sin(t * Math.PI * 2 * freqX + phaseX);
    const cosX = Math.cos(t * Math.PI * 2 * freqX + phaseX * 0.75);
    const sinZ = Math.sin(t * Math.PI * 2 * freqZ + phaseZ);
    const cosZ = Math.cos(t * Math.PI * 2 * freqZ + phaseZ * 0.75);
    const jitterNoise = randomSigned(policy.seed + index, 101, jitter * 1.1);
    const jitterNoiseZ = randomSigned(policy.seed + index + 111, 103, jitter * 1.1);
    const envelope = Math.sin(Math.PI * t);

    const offsetX = (biasX + ampX * sinX * 0.48 + ampX * cosX * 0.18 + jitterNoise) * (1 - track) * envelope;
    const offsetZ = (biasZ + ampZ * sinZ * 0.48 + ampZ * cosZ * 0.18 + jitterNoiseZ) * (1 - track) * envelope;
    const lifted = index === 0 || index === safePath.length - 1
      ? { x: point.x, z: point.z }
      : { x: point.x + offsetX, z: point.z + offsetZ };

    points.push({
      x: THREE.MathUtils.clamp(lifted.x, -1, 1),
      z: THREE.MathUtils.clamp(lifted.z, -1, 1)
    });
  }

  return points;
}

function evaluateCandidatePath(targetPath, candidatePath) {
  if (!targetPath || targetPath.length < 2 || !candidatePath || candidatePath.length < 2) {
    return {
      score: 0,
      ungatedScore: 0,
      shape: 0,
      coverage: 0,
      precision: 0,
      curve: 0,
      order: 0,
      length: 0,
      lengthMatch: 0,
      humanLength: 0,
      aiLength: 0,
      lengthRatio: 0
    };
  }

  const normalizedCandidate = resampleByArcLength(candidatePath, SWARM_EVO_EVAL_POINTS);
  const normalizedTarget = resampleByArcLength(targetPath, SWARM_EVO_EVAL_POINTS);
  const targetSegments = buildSegments(normalizedTarget);
  const candidateSegments = buildSegments(normalizedCandidate);
  const targetToCandidate = matchSegments(targetSegments, candidateSegments);
  const candidateToTarget = matchSegments(candidateSegments, targetSegments);
  const shapeScore = scoreFromError((targetToCandidate.shapeError + candidateToTarget.shapeError) * 0.5);
  const curveScore = scoreFromError((targetToCandidate.curveError + candidateToTarget.curveError) * 0.5);
  const orderScore = scoreFromError((targetToCandidate.orderError + candidateToTarget.orderError) * 0.5);
  const coverageScore = candidateToTarget.coverage;
  const precisionScore = targetToCandidate.coverage;
  const humanLength = pathLength(normalizedTarget);
  const aiLength = pathLength(normalizedCandidate);
  const lengthRatio = humanLength > 1e-6 ? aiLength / humanLength : 0;
  const lengthMatch = THREE.MathUtils.clamp(Math.min(lengthRatio, 1 / Math.max(lengthRatio, 1e-6)), 0, 1);
  const lengthMultiplier = lengthGate(lengthMatch);
  const ungatedScore =
    shapeScore * 0.34 + coverageScore * 0.24 + precisionScore * 0.18 + curveScore * 0.13 + orderScore * 0.11;
  const score = ungatedScore * lengthMultiplier;

  return {
    score,
    ungatedScore,
    shape: shapeScore,
    coverage: coverageScore,
    precision: precisionScore,
    curve: curveScore,
    order: orderScore,
    length: lengthMultiplier,
    lengthMatch,
    humanLength,
    aiLength,
    lengthRatio
  };
}

function buildPolicyPlansFromSamples(samples, count, learning = 0.45) {
  if (samples.length < 2) return [];
  const targetNorm = samples.map((sample) => ({ x: sample.target[0], z: sample.target[1] }));
  const countSafe = Math.max(0, Math.floor(count));
  const plans = [];

  for (let index = 0; index < countSafe; index += 1) {
    const policy = buildLegacySwarmPolicy(index, learning);
    const path = buildCandidatePathFromPolicy(targetNorm, policy);
    plans.push({
      policy,
      path,
      score: Math.max(0, (1 - random01(index, 30)) * 0.5),
      index
    });
  }

  return plans;
}

function updateSwarmFromPolicyPlan(plans, visibleCount = null) {
  if (!Array.isArray(plans) || plans.length === 0) {
    swarmMesh.geometry.setDrawRange(0, 0);
    return;
  }

  const visible = Math.max(0, Math.min(SWARM_AGENT_COUNT, visibleCount ?? plans.length));
  const basePath = plans[0]?.path || [];
  if (basePath.length < 2) {
    swarmMesh.geometry.setDrawRange(0, 0);
    return;
  }

  const baseSegments = basePath.length - 1;
  const positions = swarmMesh.geometry.attributes.position.array;
  const colors = swarmMesh.geometry.attributes.color.array;
  let vertexIndex = 0;

  for (let segmentIndex = 0; segmentIndex < baseSegments; segmentIndex += 1) {
    const startPointIndex = segmentIndex;
    const endPointIndex = segmentIndex + 1;
    for (let agentIndex = 0; agentIndex < visible; agentIndex += 1) {
      const plan = plans[agentIndex];
      if (!plan?.path || plan.path.length < basePath.length) continue;
      const startAgent = plan.path[startPointIndex];
      const endAgent = plan.path[endPointIndex];
      const score = Number.isFinite(plan?.score) ? plan.score : 0;
      const hueIndex = (((plan.index ?? agentIndex) % SWARM_AGENT_COUNT) + SWARM_AGENT_COUNT) * 3;
      const intensity = 0.6 + score * 0.9;
      const red = Math.min(1, SWARM_AGENT_COLORS[hueIndex] * intensity);
      const green = Math.min(1, SWARM_AGENT_COLORS[hueIndex + 1] * intensity);
      const blue = Math.min(1, SWARM_AGENT_COLORS[hueIndex + 2] * intensity);
      const startWorld = normToWorld(startAgent, PANEL.rightX);
      const endWorld = normToWorld(endAgent, PANEL.rightX);
      const startOffset = vertexIndex * 3;
      const endOffset = startOffset + 3;

      positions[startOffset] = startWorld.x;
      positions[startOffset + 1] = 0.15;
      positions[startOffset + 2] = startWorld.z;
      positions[endOffset] = endWorld.x;
      positions[endOffset + 1] = 0.15;
      positions[endOffset + 2] = endWorld.z;

      colors[startOffset] = red;
      colors[startOffset + 1] = green;
      colors[startOffset + 2] = blue;
      colors[endOffset] = red;
      colors[endOffset + 1] = green;
      colors[endOffset + 2] = blue;

      vertexIndex += 2;
    }
  }

  const fillVertexCount = baseSegments * visible * 2;
  swarmMesh.geometry.setDrawRange(0, Math.min(fillVertexCount, SWARM_VERTEX_COUNT));
  swarmMesh.geometry.attributes.position.needsUpdate = true;
  swarmMesh.geometry.attributes.color.needsUpdate = true;
}

function updateSwarmFromSamples(samples, learning = 0.45) {
  const targetCount = Math.max(0, Math.floor(state.swarmPopulationSize || SWARM_EVO_POPULATION));
  const plans = buildPolicyPlansFromSamples(samples, targetCount, learning);
  if (!plans.length) {
    swarmMesh.geometry.setDrawRange(0, 0);
    return;
  }

  const visible = Math.min(plans.length, SWARM_AGENT_COUNT);
  updateSwarmFromPolicyPlan(plans.slice(0, visible), visible);
}

function updateSwarmDrawProgress(progress) {
  const visibleSegments = Math.floor(THREE.MathUtils.clamp(progress, 0, 1) * SWARM_SEGMENTS_PER_AGENT);
  const visibleAgents = Math.max(1, Math.min(SWARM_AGENT_COUNT, state.swarmPopulationSize || SWARM_EVO_POPULATION));
  swarmMesh.geometry.setDrawRange(0, Math.min(visibleSegments * visibleAgents * 2, SWARM_VERTEX_COUNT));
}

function buildTargetLinePath(samples = state.trainingSamples) {
  if (!Array.isArray(samples) || samples.length < 2) return [];
  return resampleByArcLength(samples.map((sample) => ({ x: sample.target[0], z: sample.target[1] })), SWARM_POINTS_PER_AGENT);
}

function selectParent(population) {
  if (!population.length) return createSwarmPolicy(Math.random() * 0x7fffffff);
  const topCutoff = Math.min(population.length, Math.max(1, SWARM_EVO_PARENT_POOL));
  const bias = Math.random() ** 1.58;
  const selected = population[Math.floor(bias * topCutoff)];
  return cloneSwarmPolicy(selected.policy);
}

async function runSwarmEvolution(samples) {
  const runId = state.trainingRunId;
  const targetPath = buildTargetLinePath(samples);
  if (!targetPath.length) return null;

  const populationSize = Math.max(8, Math.min(SWARM_EVO_POPULATION, SWARM_AGENT_COUNT));
  let population = Array.from({ length: populationSize }, (_, index) =>
    cloneSwarmPolicy(createSwarmPolicy(index + runId * 997 + 1))
  );
  let bestCandidate = null;
  state.swarmPopulationSize = populationSize;
  state.swarmGenerationHistory = [];
  state.swarmGeneration = 0;
  state.swarmBestScore = null;

  for (let generation = 0; generation < SWARM_EVO_GENERATIONS; generation += 1) {
    const scored = [];
    let evaluated = 0;

    for (let batchStart = 0; batchStart < population.length; batchStart += SWARM_VIS_BATCH_SIZE) {
      if (!state.isTraining || state.trainingRunId !== runId || !state.model) return bestCandidate;

      const batchEnd = Math.min(population.length, batchStart + SWARM_VIS_BATCH_SIZE);
      const batch = population.slice(batchStart, batchEnd);
      for (let offset = 0; offset < batch.length; offset += 1) {
        const policy = batch[offset];
        const path = buildCandidatePathFromPolicy(targetPath, policy);
        const metrics = evaluateCandidatePath(targetPath, path);
        const plan = {
          policy,
          path,
          ...metrics,
          index: batchStart + offset
        };
        scored.push(plan);
        evaluated += 1;
      }

      const visible = Math.min(scored.length, SWARM_AGENT_COUNT);
      updateSwarmFromPolicyPlan(scored, visible);
      state.swarmGeneration = generation;
      state.swarmDrawProgress = ((generation + evaluated / population.length) / SWARM_EVO_GENERATIONS);
      persistRunSnapshot({ reason: "swarm-evolution-progress" });
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }

    scored.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    if (!scored.length) continue;

    const bestScore = scored[0].score ?? 0;
    const meanScore = scored.reduce((sum, item) => sum + (item.score ?? 0), 0) / scored.length;
    const averageLength = scored.reduce((sum, item) => sum + (item.lengthRatio || 0), 0) / scored.length;
    state.swarmGenerationHistory.push({
      generation: generation + 1,
      best: bestScore,
      mean: meanScore,
      averageLength
    });

    if (!state.swarmBestScore || bestScore > state.swarmBestScore) {
      state.swarmBestScore = bestScore;
      state.swarmBestAgentPolicy = cloneSwarmPolicy(scored[0].policy);
      bestCandidate = scored[0];
    } else if (bestCandidate === null && scored[0]?.score === state.swarmBestScore) {
      bestCandidate = scored[0];
    }

    state.swarmGeneration = generation + 1;

    if (generation < SWARM_EVO_GENERATIONS - 1) {
      const elites = scored.slice(0, SWARM_EVO_ELITE_COUNT).map((entry) => cloneSwarmPolicy(entry.policy));
      const nextPopulation = elites;
      while (nextPopulation.length < population.length) {
        const parentA = selectParent(scored);
        const parentB = selectParent(scored);
        nextPopulation.push(blendSwarmPolicies(parentA, parentB));
      }
      population = nextPopulation.slice(0, population.length);
    }
  }

  state.swarmDrawProgress = 1;
  if (!bestCandidate && state.swarmBestAgentPolicy) {
    const path = buildCandidatePathFromPolicy(targetPath, state.swarmBestAgentPolicy);
    bestCandidate = {
      policy: cloneSwarmPolicy(state.swarmBestAgentPolicy),
      path,
      score: state.swarmBestScore
    };
  }

  if (bestCandidate) {
    updateSwarmFromPolicyPlan([{ ...bestCandidate, index: bestCandidate.index ?? 0 }], 1);
  } else {
    updateSwarmFromPolicyPlan([], 0);
  }
  persistRunSnapshot({ reason: "swarm-evolution-complete", status: "complete" });
  return bestCandidate;
}

function convertNormToWorldPath(normPath) {
  return normPath.map((point) => normToWorld(point, PANEL.rightX));
}

function buildAiStrokeFromPolicy(policy) {
  if (!policy) return [];
  const candidatePath = buildCandidatePathFromPolicy(buildTargetLinePath(), policy);
  return convertNormToWorldPath(candidatePath);
}

function distance2D(a, b) {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

function interpolatePoint(a, b, t) {
  return {
    x: THREE.MathUtils.lerp(a.x, b.x, t),
    z: THREE.MathUtils.lerp(a.z, b.z, t)
  };
}

function pathLength(points) {
  let length = 0;
  for (let i = 1; i < points.length; i += 1) {
    length += distance2D(points[i - 1], points[i]);
  }
  return length;
}

function resampleByArcLength(points, count) {
  if (points.length === 0) return [];
  if (points.length === 1 || count <= 1) return Array.from({ length: count }, () => ({ ...points[0] }));

  const cumulative = [0];
  for (let i = 1; i < points.length; i += 1) {
    cumulative[i] = cumulative[i - 1] + distance2D(points[i - 1], points[i]);
  }

  const totalLength = cumulative[cumulative.length - 1];
  if (totalLength < 1e-6) return Array.from({ length: count }, () => ({ ...points[0] }));

  const sampled = [];
  let sourceIndex = 1;
  for (let i = 0; i < count; i += 1) {
    const targetDistance = (i / (count - 1)) * totalLength;
    while (sourceIndex < cumulative.length - 1 && cumulative[sourceIndex] < targetDistance) {
      sourceIndex += 1;
    }
    const previousDistance = cumulative[sourceIndex - 1];
    const segmentLength = cumulative[sourceIndex] - previousDistance || 1;
    const t = (targetDistance - previousDistance) / segmentLength;
    sampled.push(interpolatePoint(points[sourceIndex - 1], points[sourceIndex], t));
  }
  return sampled;
}

function angleBetween(a, b) {
  const dot = THREE.MathUtils.clamp(a.x * b.x + a.z * b.z, -1, 1);
  return Math.acos(dot) / Math.PI;
}

function buildSegments(points) {
  const segments = [];
  for (let i = 1; i < points.length; i += 1) {
    const start = points[i - 1];
    const end = points[i];
    const dx = end.x - start.x;
    const dz = end.z - start.z;
    const length = Math.hypot(dx, dz);
    if (length < 1e-6) continue;
    segments.push({
      index: segments.length,
      t: (i - 1) / Math.max(1, points.length - 2),
      mid: { x: (start.x + end.x) * 0.5, z: (start.z + end.z) * 0.5 },
      dir: { x: dx / length, z: dz / length },
      length,
      curve: 0
    });
  }

  for (let i = 0; i < segments.length; i += 1) {
    const prev = segments[Math.max(0, i - 1)];
    const next = segments[Math.min(segments.length - 1, i + 1)];
    segments[i].curve = angleBetween(prev.dir, next.dir);
  }

  return segments;
}

function segmentCost(source, target) {
  const spatial = distance2D(source.mid, target.mid) / MATCH_DISTANCE;
  const direction = angleBetween(source.dir, target.dir);
  const curve = Math.abs(source.curve - target.curve);
  const order = Math.abs(source.t - target.t);
  return {
    spatial,
    direction,
    curve,
    order,
    total: spatial + direction * 0.65 + curve * 0.45 + order * 0.35
  };
}

function matchSegments(sourceSegments, targetSegments) {
  if (!sourceSegments.length || !targetSegments.length) {
    return { coverage: 0, shapeError: 1, orderError: 1, curveError: 1 };
  }

  let matched = 0;
  let totalShape = 0;
  let totalOrder = 0;
  let totalCurve = 0;

  for (const source of sourceSegments) {
    let best = null;
    for (const target of targetSegments) {
      const cost = segmentCost(source, target);
      if (!best || cost.total < best.total) best = cost;
    }

    if (best.spatial <= 1.25 && best.direction <= 0.45) matched += 1;
    totalShape += Math.min(2, best.spatial + best.direction * 0.75);
    totalOrder += Math.min(1, best.order);
    totalCurve += Math.min(1, best.curve);
  }

  const denom = sourceSegments.length;
  return {
    coverage: matched / denom,
    shapeError: totalShape / denom / 2,
    orderError: totalOrder / denom,
    curveError: totalCurve / denom
  };
}

function scoreFromError(error) {
  return THREE.MathUtils.clamp(1 - error, 0, 1);
}

function lengthGate(lengthScore) {
  if (lengthScore <= 0.8) return 0;
  const normalized = (lengthScore - 0.8) / 0.2;
  return normalized * normalized;
}

function evaluateStroke(humanWorld, aiWorld) {
  if (humanWorld.length < 2 || aiWorld.length < 2) return null;

  const humanNorm = humanWorld.map((point) => worldToNorm(point, PANEL.leftX));
  const aiNorm = aiWorld.map((point) => worldToNorm(point, PANEL.rightX));
  const humanLength = pathLength(humanNorm);
  const aiLength = pathLength(aiNorm);
  const lengthRatio = humanLength > 1e-6 ? aiLength / humanLength : 0;
  const lengthScore = THREE.MathUtils.clamp(
    Math.min(lengthRatio, 1 / Math.max(lengthRatio, 1e-6)),
    0,
    1
  );

  const humanSampled = resampleByArcLength(humanNorm, EVAL_POINTS);
  const aiSampled = resampleByArcLength(aiNorm, EVAL_POINTS);
  const humanSegments = buildSegments(humanSampled);
  const aiSegments = buildSegments(aiSampled);
  const humanToAi = matchSegments(humanSegments, aiSegments);
  const aiToHuman = matchSegments(aiSegments, humanSegments);

  const shapeScore = scoreFromError((humanToAi.shapeError + aiToHuman.shapeError) * 0.5);
  const curveScore = scoreFromError((humanToAi.curveError + aiToHuman.curveError) * 0.5);
  const orderScore = scoreFromError((humanToAi.orderError + aiToHuman.orderError) * 0.5);
  const coverageScore = humanToAi.coverage;
  const precisionScore = aiToHuman.coverage;
  const ungatedScore =
    shapeScore * 0.3 +
    coverageScore * 0.23 +
    precisionScore * 0.17 +
    curveScore * 0.14 +
    orderScore * 0.16;
  const lengthMultiplier = lengthGate(lengthScore);
  const finalScore = ungatedScore * lengthMultiplier;

  return {
    score: finalScore,
    ungatedScore,
    shape: shapeScore,
    coverage: coverageScore,
    precision: precisionScore,
    curve: curveScore,
    order: orderScore,
    length: lengthMultiplier,
    lengthMatch: lengthScore,
    humanLength,
    aiLength,
    lengthRatio
  };
}

async function trainFromStroke({ extra = false } = {}) {
  if (state.rawStroke.length < 2) {
    updateUi("Draw a little longer, then release.");
    return;
  }

  if (extra && (!state.model || state.trainingSamples.length === 0)) {
    updateUi("Draw once before training more.");
    return;
  }

  state.isTraining = true;
  state.isReplaying = false;
  state.aiStroke = [];
  state.evaluation = null;
  updateLine(aiLine, []);
  updateLine(aiGhost, []);
  updateLine(humanGhost, state.rawStroke);
  const continuing = extra || (keepLearning.checked && state.model);
  updateUi(continuing ? "Continuing training..." : "Training...");

  const samples = extra ? state.trainingSamples : buildTrainingSamples(state.rawStroke);
  state.trainingSamples = samples;
  updateLine(targetLine, trainingSamplesToWorld(samples));
  updateSwarmFromSamples(samples, continuing ? 0.62 : 0.36);
  updateSwarmDrawProgress(0);
  const model = continuing ? state.model : new StrokeNet(DEFAULT_MODEL_HIDDEN_SIZE);
  state.model = model;
  if (!continuing) state.epochs = 0;
  state.loss = null;

  const totalEpochs = extra ? EXTRA_TRAIN_EPOCHS : BASE_TRAIN_EPOCHS;
  const runId = state.trainingRunId + 1;
  const startingEpochs = state.epochs;
  state.trainingRunId = runId;
  state.swarmDrawProgress = 0;
  state.swarmGeneration = 0;
  state.swarmGenerationHistory = [];
  let trained = 0;

  while (trained < totalEpochs && state.model === model && state.trainingRunId === runId) {
    const trainBlock = Math.min(TRAIN_CHUNK_SIZE * TRAIN_LOOP_YIELD_CHUNKS, totalEpochs - trained);
    for (let loop = 0; loop < trainBlock; loop += TRAIN_CHUNK_SIZE) {
      const chunk = Math.min(TRAIN_CHUNK_SIZE, totalEpochs - trained);
      state.loss = model.train(
        samples,
        chunk,
        LEARNING_RATE,
        (epoch, loss) => {
          state.epochs = startingEpochs + trained + epoch;
          state.loss = loss;
        }
      );
      trained += chunk;
      state.swarmDrawProgress = trained / totalEpochs;
      updateSwarmDrawProgress(state.swarmDrawProgress);
      persistRunSnapshot({ reason: "training-progress" });

      if (state.model !== model || state.trainingRunId !== runId) break;
    }
    pulseNeuralCloud();
    updateUi(continuing ? "Continuing training..." : "Training...");
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  if (state.model !== model || state.trainingRunId !== runId) {
    state.isTraining = false;
    return;
  }
  state.isTraining = false;
  updateUi(continuing ? "Running swarm evolution..." : "Running swarm evolution...");
  updateSwarmFromSamples(samples, Math.min(0.94, 0.35 + state.epochs / 1400));
  const compactResult = trainDistillModel(model, samples, {
    stride: TINY_DISTILL_STRIDE,
    learningRate: TINY_DISTILL_LR,
    maxSamples: TINY_DISTILL_BATCH * 8
  });
  if (compactResult && compactResult.model) {
    state.loss = compactResult.loss;
    state.epochs += compactResult.epochCount;
    state.trainingSamples = samples;
  }
  persistRunSnapshot({
    reason: "training-complete",
    status: "complete",
    compactSampleCount: compactResult?.sampleCount ?? null,
    compactEpochs: compactResult?.epochCount ?? null
  });
  state.isTraining = true;
  const bestCandidate = await runSwarmEvolution(samples);
  if (bestCandidate?.path) {
    const candidateStroke = convertNormToWorldPath(bestCandidate.path);
    evaluateAiStroke(candidateStroke, "swarm");
  }
  state.isTraining = false;
  generateAiStroke(state.model || model);
  if (state.bestAiStroke.length > 0) {
    state.aiStroke = state.bestAiStroke.slice();
  }
  startReplay();
}

function evaluateAiStroke(candidateStroke, source = "model") {
  if (!Array.isArray(candidateStroke) || candidateStroke.length < 2) {
    return {
      stroke: [],
      score: null,
      accepted: false,
      evaluation: null
    };
  }

  const evaluation = evaluateStroke(state.rawStroke, candidateStroke);
  const score = evaluation ? evaluation.score : null;
  const candidateIsBetter = !Number.isFinite(state.bestAiScore) || (Number.isFinite(score) && score > state.bestAiScore);
  const accepted = candidateIsBetter;

  if (candidateIsBetter) {
    state.bestAiStroke = candidateStroke.map((point) => point.clone());
    state.bestAiEvaluation = evaluation;
    state.bestAiScore = Number.isFinite(score) ? score : state.bestAiScore;
    state.bestAiSource = source;
    state.evaluation = evaluation;
    updateLine(aiGhost, state.bestAiStroke);
    state.aiStroke = state.bestAiStroke.slice();
  } else {
    state.evaluation = state.evaluation ?? evaluation;
  }

  return {
    stroke: candidateStroke,
    score,
    accepted,
    evaluation
  };
}

function generateAiStroke(model) {
  if (model) {
    const candidate = [];
    const pointCount = Math.max(16, Math.min(REPLAY_POINTS, state.rawStroke.length));
    for (let i = 0; i < pointCount; i += 1) {
      const t = pointCount === 1 ? 0 : i / (pointCount - 1);
      const { output } = model.forward(features(t));
      candidate.push(normToWorld({ x: output[0], z: output[1] }, PANEL.rightX));
    }
    state.aiStroke = candidate;
  }
  if (!state.aiStroke.length) {
    return {
      stroke: [],
      score: null,
      accepted: false,
      evaluation: null
    };
  }
  return evaluateAiStroke(state.aiStroke, "model");
}

function applyAiStrokeFromPolicy(policy) {
  state.aiStroke = buildAiStrokeFromPolicy(policy);
  return evaluateAiStroke(state.aiStroke, "swarm");
}

function startReplay() {
  const sourceStroke = state.bestAiStroke.length > 0 ? state.bestAiStroke : state.aiStroke;
  if (!sourceStroke.length) return;
  state.aiStroke = sourceStroke.slice();
  state.isReplaying = true;
  state.replayCursor = 0;
  state.replayStartedAt = performance.now();
  updateLine(aiLine, []);
  updateSwarmDrawProgress(0);
  aiCursor.position.copy(state.aiStroke[0]);
  const sourceLabel = state.bestAiSource || "model";
  updateUi(`AI is repeating ${sourceLabel} line.`);
}

function scheduleAutoReplay() {
  if (!repeatBest || !repeatBest.checked || !state.bestAiStroke.length) return;
  const runToken = state.trainingRunId;
  setTimeout(() => {
    if (
      !state.isReplaying &&
      !state.isTraining &&
      !state.isDrawing &&
      repeatBest &&
      repeatBest.checked &&
      state.trainingRunId === runToken
    ) {
      startReplay();
    }
  }, AUTO_REPLAY_DELAY_MS);
}

function updateReplay(now) {
  if (!state.isReplaying || state.aiStroke.length < 2) return;
  const elapsed = now - state.replayStartedAt;
  const duration = Math.max(700, Math.min(4200, state.rawStroke.length * 10));
  const progress = THREE.MathUtils.clamp(elapsed / duration, 0, 1);
  const nextCursor = Math.floor(progress * (state.aiStroke.length - 1));
  updateSwarmDrawProgress(progress);

  if (nextCursor !== state.replayCursor) {
    state.replayCursor = nextCursor;
    const visible = state.aiStroke.slice(0, state.replayCursor + 1);
    updateLine(aiLine, visible);
    aiCursor.position.copy(state.aiStroke[state.replayCursor]);
  }

  if (progress >= 1) {
    state.isReplaying = false;
    const finalStroke = state.bestAiStroke.length > 0 ? state.bestAiStroke : state.aiStroke;
    if (finalStroke.length > 0) {
      state.aiStroke = finalStroke.slice();
      updateLine(aiLine, finalStroke);
      aiCursor.position.copy(finalStroke[finalStroke.length - 1]);
    }
    updateSwarmDrawProgress(1);
    const score = state.evaluation ? Math.round(state.evaluation.score * 100) : null;
    const nextStep = keepLearning.checked ? "Train more or draw again." : "Draw again to retrain.";
    updateUi(score === null ? `Done. ${nextStep}` : `Score ${score}. ${nextStep}`);
    if (repeatBest && repeatBest.checked) {
      scheduleAutoReplay();
    }
  }
}

function updateUi(status) {
  if (status) runStatus.textContent = status;
  const evaluation = state.evaluation;
  scoreValue.textContent = evaluation ? Math.round(evaluation.score * 100) : "--";
  lossValue.textContent = state.loss === null ? "--" : state.loss.toFixed(4);
  pointValue.textContent = state.rawStroke.length.toLocaleString();
  epochValue.textContent = state.epochs.toLocaleString();
  coverageValue.textContent = evaluation ? `${Math.round(evaluation.coverage * 100)}%` : "--";
  precisionValue.textContent = evaluation ? `${Math.round(evaluation.precision * 100)}%` : "--";
  curveValue.textContent = evaluation ? `${Math.round(evaluation.curve * 100)}%` : "--";
  lengthValue.textContent = evaluation ? `${Math.round(evaluation.length * 100)}%` : "--";
  orderValue.textContent = evaluation ? `${Math.round(evaluation.order * 100)}%` : "--";
  agentValue.textContent = (state.swarmPopulationSize || SWARM_EVO_POPULATION).toLocaleString();
  generationValue.textContent = `${state.swarmGeneration}/${SWARM_EVO_GENERATIONS}`;
  trainMoreButton.disabled = state.isTraining || state.rawStroke.length < 2 || !state.model;
}

function pulseNeuralCloud() {
  neuralCloud.children.forEach((node) => {
    node.scale.setScalar(1.2 + Math.random() * 1.8);
  });
}

function onPointerDown(event) {
  event.preventDefault();
  const point = eventToLeftPoint(event);
  if (!point) {
    updateUi("Start on the left grid.");
    return;
  }
  canvas.setPointerCapture(event.pointerId);
  state.isDrawing = true;
  state.isTraining = false;
  state.isReplaying = false;
  state.trainingRunId += 1;
  clearPersistedRun();
  if (!keepLearning.checked) {
    state.model = null;
    state.compactModel = null;
    state.trainingSamples = [];
    state.epochs = 0;
  }
  state.rawStroke = [];
  state.aiStroke = [];
  state.bestAiStroke = [];
  state.bestAiScore = null;
  state.bestAiEvaluation = null;
  state.bestAiSource = null;
  state.evaluation = null;
  state.loss = null;
  updateLine(aiLine, []);
  updateLine(aiGhost, []);
  updateLine(targetLine, []);
  updateLine(humanGhost, []);
  swarmMesh.geometry.setDrawRange(0, 0);
  aiCursor.position.set(-999, 0, -999);
  addStrokePoint(point);
  updateUi("Drawing...");
}

function onPointerMove(event) {
  event.preventDefault();
  if (!state.isDrawing) return;
  const events = typeof event.getCoalescedEvents === "function" ? event.getCoalescedEvents() : [event];
  for (const coalescedEvent of events) {
    const point = eventToLeftPoint(coalescedEvent);
    if (point) addStrokePoint(point);
  }
}

function onPointerUp(event) {
  event.preventDefault();
  if (!state.isDrawing) return;
  const point = eventToLeftPoint(event);
  if (point) addStrokePoint(point);
  state.isDrawing = false;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  trainFromStroke();
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const isMobile = width < 760;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  stage.scale.setScalar(isMobile ? 0.72 : 1);
  stage.position.z = isMobile ? -0.8 : 0;
  camera.position.set(0, isMobile ? 11.2 : 8.8, isMobile ? 13.1 : 10.2);
  camera.lookAt(0, 0, isMobile ? -0.65 : 0);
  camera.updateProjectionMatrix();
}

function animate(now) {
  renderFrame += 1;
  requestAnimationFrame(animate);
  updateReplay(now);
  neuralCloud.children.forEach((node, index) => {
    node.position.y += Math.sin(now * 0.002 + node.userData.phase) * 0.0009;
    node.scale.lerp(new THREE.Vector3(1, 1, 1), 0.045);
    node.rotation.y += 0.006 + index * 0.00008;
  });
  renderer.render(scene, camera);
}

canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("pointercancel", onPointerUp);
trainMoreButton.addEventListener("click", () => trainFromStroke({ extra: true }));
keepLearning.addEventListener("change", () => {
  updateUi(keepLearning.checked ? "Keep learning is on." : "Keep learning is off. Next drawing starts fresh.");
});
resetButton.addEventListener("click", clearStroke);
window.addEventListener("resize", resize);

resize();
if (!loadPersistedRun()) {
  clearStroke();
} else if (!state.model) {
  updateUi("Loaded previous drawing.");
}

requestAnimationFrame(animate);

window.__mntDebug = {
  getState() {
    return {
      isDrawing: state.isDrawing,
      isTraining: state.isTraining,
      isReplaying: state.isReplaying,
      isRestoredRun: state.isRestoredRun,
      isPersisted: Boolean(localStorage.getItem(RUN_STORAGE_KEY)),
      rawStrokeCount: state.rawStroke.length,
      sampleCount: state.trainingSamples.length,
      aiStrokeCount: state.aiStroke.length,
      epochs: state.epochs,
      loss: state.loss,
      evaluationScore: state.evaluation?.score ?? null,
      coverage: state.evaluation?.coverage ?? null,
      precision: state.evaluation?.precision ?? null,
      curve: state.evaluation?.curve ?? null,
      order: state.evaluation?.order ?? null,
      length: state.evaluation?.length ?? null,
      bestAiScore: state.bestAiScore,
      bestAiSource: state.bestAiSource,
      bestAiStrokeCount: state.bestAiStroke.length,
      repeatBest: repeatBest?.checked ?? null,
      targetLinePointCount: targetLine.geometry.attributes.position.count,
      swarmProgress: state.swarmDrawProgress,
      swarmGeneration: state.swarmGeneration,
      swarmBestScore: state.swarmBestScore,
      swarmPopulationSize: state.swarmPopulationSize,
      swarmGenerationHistory: state.swarmGenerationHistory,
      swarmRenderRange: swarmMesh.geometry.drawRange.count,
      swarmRenderOrder: swarmMesh.renderOrder,
      targetRenderOrder: targetLine.renderOrder,
      targetDepthTest: targetLine.material.depthTest,
      targetDepthWrite: targetLine.material.depthWrite,
      compactModelMeta: null,
      keepLearning: keepLearning.checked,
      renderFrame
    };
  },
  projectToCanvas(worldX, worldY, worldZ) {
    const point = new THREE.Vector3(worldX, worldY, worldZ).project(camera);
    return {
      x: (point.x * 0.5 + 0.5) * canvas.clientWidth,
      y: (-point.y * 0.5 + 0.5) * canvas.clientHeight,
      z: point.z
    };
  },
  leftFromNorm(xNorm, zNorm) {
    const world = normToWorld({ x: xNorm, z: zNorm }, PANEL.leftX);
    return {
      x: world.x,
      y: world.y,
      z: world.z
    };
  }
};
