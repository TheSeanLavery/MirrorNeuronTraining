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
const LEARNING_RATE = 0.024;
const SWARM_AGENT_COUNT = 1200;
const SWARM_SEGMENTS = 32;
const SWARM_INSTANCE_COUNT = SWARM_AGENT_COUNT * SWARM_SEGMENTS;

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
const swarmMatrix = new THREE.Matrix4();
const swarmXAxis = new THREE.Vector3();
const swarmYAxis = new THREE.Vector3();
const swarmZAxis = new THREE.Vector3(0, 1, 0);

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
  swarm: new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0.46,
    vertexColors: true,
    depthWrite: false,
    side: THREE.DoubleSide
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
  rawStroke: [],
  trainStroke: [],
  aiStroke: [],
  replayCursor: 0,
  replayStartedAt: 0,
  loss: null,
  epochs: 0,
  evaluation: null,
  model: null,
  trainingSamples: [],
  trainingRunId: 0,
  swarmGeneration: 0
};

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
const keepLearning = document.querySelector("#keepLearning");
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
  const geometry = new THREE.PlaneGeometry(1, 1);
  const mesh = new THREE.InstancedMesh(geometry, materials.swarm, SWARM_INSTANCE_COUNT);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.count = 0;
  mesh.frustumCulled = false;
  stage.add(mesh);
  return mesh;
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
      if (epoch % 20 === 0 || epoch === epochs - 1) onProgress(epoch + 1, finalLoss);
    }
    return finalLoss;
  }
}

function randomMatrix(rows, cols, scale) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
  );
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
  state.trainingSamples = [];
  humanCursor.position.set(-999, 0, -999);
  aiCursor.position.set(-999, 0, -999);
  updateLine(humanLine, []);
  updateLine(aiLine, []);
  updateLine(targetLine, []);
  updateLine(humanGhost, []);
  updateLine(aiGhost, []);
  swarmMesh.count = 0;
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

function hashUnit(agentIndex, salt) {
  const value = Math.sin((agentIndex + 1) * 127.1 + salt * 311.7) * 43758.5453;
  return value - Math.floor(value);
}

function agentPointFromTarget(point, agentIndex, t, learning) {
  const quality = Math.pow(hashUnit(agentIndex, 4), 0.42);
  const phaseA = hashUnit(agentIndex, 1) * Math.PI * 2;
  const phaseB = hashUnit(agentIndex, 2) * Math.PI * 2;
  const biasX = (hashUnit(agentIndex, 5) - 1) * 0.22;
  const biasZ = (hashUnit(agentIndex, 6) - 1) * 0.22;
  const envelope = Math.sin(Math.PI * t);
  const amplitude = (0.03 + (1 - quality) * 0.32) * (1 - learning * 0.72);
  const wobbleX = Math.sin(t * Math.PI * 2 + phaseA) * amplitude * envelope;
  const wobbleZ = Math.cos(t * Math.PI * 3 + phaseB) * amplitude * envelope;

  return {
    x: THREE.MathUtils.clamp(point.x + wobbleX + biasX * amplitude, -1, 1),
    z: THREE.MathUtils.clamp(point.z + wobbleZ + biasZ * amplitude, -1, 1),
    quality
  };
}

function setSegmentInstance(instanceIndex, start, end, color, thickness) {
  const dx = end.x - start.x;
  const dz = end.z - start.z;
  const length = Math.hypot(dx, dz);
  if (length < 1e-5) return false;

  swarmXAxis.set(dx / length, 0, dz / length).multiplyScalar(length);
  swarmYAxis.set(-dz / length, 0, dx / length).multiplyScalar(thickness);
  swarmMatrix.makeBasis(swarmXAxis, swarmYAxis, swarmZAxis);
  swarmMatrix.setPosition((start.x + end.x) * 0.5, 0.102, (start.z + end.z) * 0.5);
  swarmMesh.setMatrixAt(instanceIndex, swarmMatrix);
  swarmMesh.setColorAt(instanceIndex, color);
  return true;
}

function updateSwarmFromSamples(samples, learning = 0.45) {
  if (samples.length < 2) {
    swarmMesh.count = 0;
    return;
  }

  const targetNorm = samples.map((sample) => ({ x: sample.target[0], z: sample.target[1] }));
  const basePath = resampleByArcLength(targetNorm, SWARM_SEGMENTS + 1);
  const color = new THREE.Color();
  let instanceIndex = 0;

  for (let segmentIndex = 0; segmentIndex < SWARM_SEGMENTS; segmentIndex += 1) {
    const t0 = segmentIndex / SWARM_SEGMENTS;
    const t1 = (segmentIndex + 1) / SWARM_SEGMENTS;

    for (let agentIndex = 0; agentIndex < SWARM_AGENT_COUNT; agentIndex += 1) {
      const startAgent = agentPointFromTarget(basePath[segmentIndex], agentIndex, t0, learning);
      const endAgent = agentPointFromTarget(basePath[segmentIndex + 1], agentIndex, t1, learning);
      const quality = (startAgent.quality + endAgent.quality) * 0.5;
      color.setHSL(0.47 + quality * 0.13, 0.76, 0.32 + quality * 0.25);
      const start = normToWorld(startAgent, PANEL.rightX);
      const end = normToWorld(endAgent, PANEL.rightX);
      setSegmentInstance(instanceIndex, start, end, color, 0.012 + quality * 0.01);
      instanceIndex += 1;
    }
  }

  swarmMesh.count = 0;
  swarmMesh.instanceMatrix.needsUpdate = true;
  if (swarmMesh.instanceColor) swarmMesh.instanceColor.needsUpdate = true;
  state.swarmGeneration += 1;
}

function updateSwarmDrawProgress(progress) {
  const visibleSegments = Math.floor(THREE.MathUtils.clamp(progress, 0, 1) * SWARM_SEGMENTS);
  swarmMesh.count = visibleSegments * SWARM_AGENT_COUNT;
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
  updateSwarmDrawProgress(1);
  const model = continuing ? state.model : new StrokeNet(96);
  state.model = model;
  if (!continuing) state.epochs = 0;
  state.loss = null;

  const totalEpochs = extra ? EXTRA_TRAIN_EPOCHS : BASE_TRAIN_EPOCHS;
  const runId = state.trainingRunId + 1;
  const startingEpochs = state.epochs;
  state.trainingRunId = runId;
  let trained = 0;

  while (trained < totalEpochs && state.model === model && state.trainingRunId === runId) {
    const chunk = Math.min(TRAIN_CHUNK_SIZE, totalEpochs - trained);
    state.loss = model.train(samples, chunk, LEARNING_RATE, (epoch, loss) => {
      state.epochs = startingEpochs + trained + epoch;
      state.loss = loss;
    });
    trained += chunk;
    pulseNeuralCloud();
    updateUi(continuing ? "Continuing training..." : "Training...");
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  if (state.model !== model || state.trainingRunId !== runId) return;
  state.isTraining = false;
  updateSwarmFromSamples(samples, Math.min(0.94, 0.35 + state.epochs / 1400));
  generateAiStroke(model);
  startReplay();
}

function generateAiStroke(model) {
  state.aiStroke = [];
  const pointCount = Math.max(16, Math.min(REPLAY_POINTS, state.rawStroke.length));
  for (let i = 0; i < pointCount; i += 1) {
    const t = pointCount === 1 ? 0 : i / (pointCount - 1);
    const { output } = model.forward(features(t));
    state.aiStroke.push(normToWorld({ x: output[0], z: output[1] }, PANEL.rightX));
  }
  state.evaluation = evaluateStroke(state.rawStroke, state.aiStroke);
  updateLine(aiGhost, state.aiStroke);
}

function startReplay() {
  state.isReplaying = true;
  state.replayCursor = 0;
  state.replayStartedAt = performance.now();
  updateLine(aiLine, []);
  updateSwarmDrawProgress(0);
  aiCursor.position.copy(state.aiStroke[0]);
  updateUi("AI is copying the target overlay.");
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
    updateLine(aiLine, state.aiStroke);
    updateSwarmDrawProgress(1);
    const score = state.evaluation ? Math.round(state.evaluation.score * 100) : null;
    const nextStep = keepLearning.checked ? "Train more or draw again." : "Draw again to retrain.";
    updateUi(score === null ? `Done. ${nextStep}` : `Score ${score}. ${nextStep}`);
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
  agentValue.textContent = SWARM_AGENT_COUNT.toLocaleString();
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
  if (!keepLearning.checked) {
    state.model = null;
    state.trainingSamples = [];
    state.epochs = 0;
  }
  state.rawStroke = [];
  state.aiStroke = [];
  state.evaluation = null;
  state.loss = null;
  updateLine(aiLine, []);
  updateLine(aiGhost, []);
  updateLine(targetLine, []);
  updateLine(humanGhost, []);
  swarmMesh.count = 0;
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
clearStroke();
requestAnimationFrame(animate);
