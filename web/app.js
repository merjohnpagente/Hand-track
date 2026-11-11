import {
  FilesetResolver,
  HandLandmarker,
  ImageSegmenter,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MP_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MP_WASM = `${MP_BASE}/wasm`;

const FILTERS = [
  "MONO",
  "DUAL-TONE",
  "PIXELATE",
  "INVERT",
  "SEPIA",
  "BLUR",
  "THERMAL",
  "SKETCH",
  "GLITCH",
  "NEON",
  "GALAXY",
];

const FINGER_TIPS = [4, 8]; // thumb tip, index tip
const GESTURE_RATIO = 0.06; // ~40 px on a 720p frame, same feel as the Python app

const $ = (id) => document.getElementById(id);
const canvas = $("stage");
const ctx = canvas.getContext("2d");
const video = $("cam");
const overlay = $("overlay");
const startBtn = $("start-btn");
const statusEl = $("status");
const hud = $("hud");
const filterNameEl = $("filter-name");
const nextBtn = $("next-filter");

// Reusable buffers: ROI-sized processing keeps filters fast on any device.
const roiCanvas = document.createElement("canvas");
const roiCtx = roiCanvas.getContext("2d", { willReadFrequently: true });
const tmpCanvas = document.createElement("canvas");
const tmpCtx = tmpCanvas.getContext("2d", { willReadFrequently: true });

let handLandmarker = null;
let segmenter = null;
let running = false;
let cameraStream = null;
let currentFilter = 0;
let gestureTriggered = false;
let lastVideoTime = -1;
let galaxy = null; // { canvas, ctx }
let personMask = null; // { data, width, height }

const JET = buildJetLut();

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

startBtn.addEventListener("click", start);
nextBtn.addEventListener("click", () => {
  cycleFilter();
});

window.addEventListener("pagehide", () => {
  running = false;
  cameraStream?.getTracks().forEach((track) => track.stop());
});

async function start() {
  startBtn.disabled = true;
  try {
    if (!window.isSecureContext) {
      throw new Error("Camera requires HTTPS or localhost.");
    }
    if (!handLandmarker) {
      await initModels();
    }
    await startCamera();

    setStatus("");
    overlay.classList.add("is-hidden");
    hud.classList.remove("is-hidden");
    nextBtn.classList.remove("is-hidden");

    running = true;
    lastVideoTime = -1;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    setStatus(`Failed: ${err.message || err}`, true);
    startBtn.disabled = false;
  }
}

async function initModels() {
  setStatus("Loading MediaPipe (WASM)…");
  const vision = await FilesetResolver.forVisionTasks(MP_WASM);

  const [handModel, segModel] = await Promise.all([
    resolveModel("hand_landmarker.task"),
    resolveModel("selfie_segmenter.tflite"),
  ]);

  setStatus("Loading hand tracking model…");
  handLandmarker = await createWithFallback((delegate) =>
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: handModel, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.7,
      minHandPresenceConfidence: 0.7,
      minTrackingConfidence: 0.7,
    }),
  );

  setStatus("Loading segmentation model (GALAXY)…");
  try {
    segmenter = await createWithFallback((delegate) =>
      ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: segModel, delegate },
        runningMode: "VIDEO",
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      }),
    );
  } catch (err) {
    // Galaxy filter just falls back to "no effect" when this model fails.
    console.warn("Segmenter unavailable:", err);
    segmenter = null;
  }
}

async function createWithFallback(create) {
  try {
    return await create("GPU");
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU.", err);
    return create("CPU");
  }
}

// Served from the repo root on Pages, or from ../ during local development
// (`python3 -m http.server` from the repo root, open /web/).
async function resolveModel(name) {
  const candidates = [
    new URL(`./${name}`, location.href),
    new URL(`../${name}`, location.href),
  ];
  for (const url of candidates) {
    try {
      const res = await fetch(url, { method: "HEAD", cache: "no-store" });
      if (res.ok) return url.href;
    } catch {
      // try the next candidate
    }
  }
  return new URL(name, location.href).href;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser does not support camera access.");
  }

  setStatus("Requesting camera permission…");
  cameraStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });

  video.srcObject = cameraStream;
  await video.play();
  if (!video.videoWidth) {
    await new Promise((resolve) => {
      video.addEventListener("loadedmetadata", resolve, { once: true });
    });
  }

  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  buildGalaxy(canvas.width, canvas.height);
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
  statusEl.style.display = text ? "" : "none";
}

/* ------------------------------------------------------------------ */
/* Render loop                                                         */
/* ------------------------------------------------------------------ */

function loop() {
  if (!running) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      render(performance.now());
    } catch (err) {
      console.warn("Skipped frame:", err);
    }
  }

  requestAnimationFrame(loop);
}

function render(timestamp) {
  const w = canvas.width;
  const h = canvas.height;

  // Camera feed, mirrored like cv2.flip(img, 1).
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  let result;
  try {
    result = handLandmarker.detectForVideo(video, timestamp);
  } catch (err) {
    console.warn("Hand detection failed:", err);
    return;
  }

  const landmarks = result.landmarks || [];
  const activeFilter = FILTERS[currentFilter];

  // The segmenter is the heaviest model, so only run it for GALAXY.
  personMask = null;
  if (activeFilter === "GALAXY" && segmenter) {
    try {
      const seg = segmenter.segmentForVideo(video, timestamp);
      if (seg.categoryMask) {
        personMask = {
          data: seg.categoryMask.getAsUint8Array(),
          width: seg.categoryMask.width,
          height: seg.categoryMask.height,
        };
      }
      seg.close?.();
    } catch (err) {
      console.warn("Segmentation failed:", err);
    }
  }

  updateGesture(landmarks, w, h);

  const portalPoints = [];
  for (const hand of landmarks) {
    for (const id of FINGER_TIPS) {
      const point = toCanvasPoint(hand[id], w, h);
      portalPoints.push(point);
      drawFingerDot(point);
    }
  }

  if (portalPoints.length === 4) {
    drawPortal(portalPoints, w, h, FILTERS[currentFilter]);
  }
}

/* ------------------------------------------------------------------ */
/* Gestures                                                            */
/* ------------------------------------------------------------------ */

function updateGesture(landmarks, w, h) {
  const threshold = Math.min(w, h) * GESTURE_RATIO;
  let changeFilter = false;

  // Index fingertips of both hands touching.
  if (landmarks.length >= 2) {
    const left = toCanvasPoint(landmarks[0][8], w, h);
    const right = toCanvasPoint(landmarks[1][8], w, h);
    if (distance(left, right) < threshold) changeFilter = true;
  }

  // Thumb tip touching the pinky tip (per hand).
  for (const hand of landmarks) {
    const thumb = toCanvasPoint(hand[4], w, h);
    const pinky = toCanvasPoint(hand[20], w, h);
    if (distance(thumb, pinky) < threshold) changeFilter = true;
  }

  if (changeFilter && !gestureTriggered) {
    gestureTriggered = true;
    cycleFilter();
  } else if (!changeFilter) {
    gestureTriggered = false;
  }
}

function cycleFilter() {
  currentFilter = (currentFilter + 1) % FILTERS.length;
  filterNameEl.textContent = FILTERS[currentFilter];
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function toCanvasPoint(landmark, w, h) {
  return { x: (1 - landmark.x) * w, y: landmark.y * h };
}

/* ------------------------------------------------------------------ */
/* Portal drawing                                                      */
/* ------------------------------------------------------------------ */

function drawPortal(points, w, h, filterName) {
  const polygon = portalPolygon(points);
  if (polygon.length !== 4) return;

  const xs = polygon.map((p) => p.x);
  const ys = polygon.map((p) => p.y);
  const x = Math.max(0, Math.floor(Math.min(...xs)));
  const y = Math.max(0, Math.floor(Math.min(...ys)));
  const x2 = Math.min(w, Math.ceil(Math.max(...xs)));
  const y2 = Math.min(h, Math.ceil(Math.max(...ys)));
  const bw = x2 - x;
  const bh = y2 - y;

  if (bw < 2 || bh < 2) return;

  const filtered = renderFilteredRoi(x, y, bw, bh, filterName);

  // Clip the filtered rectangle to the finger portal.
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(polygon[0].x, polygon[0].y);
  for (let i = 1; i < polygon.length; i++) {
    ctx.lineTo(polygon[i].x, polygon[i].y);
  }
  ctx.closePath();
  ctx.clip();
  ctx.drawImage(filtered, x, y);
  ctx.restore();

  // White outline.
  ctx.beginPath();
  ctx.moveTo(polygon[0].x, polygon[0].y);
  for (let i = 1; i < polygon.length; i++) {
    ctx.lineTo(polygon[i].x, polygon[i].y);
  }
  ctx.closePath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
  ctx.stroke();

  // Glowing particles along the edges.
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    for (let k = 0; k < 5; k++) {
      const t = Math.random();
      const px = a.x * t + b.x * (1 - t) + (Math.random() - 0.5) * 30;
      const py = a.y * t + b.y * (1 - t) + (Math.random() - 0.5) * 30;
      ctx.beginPath();
      ctx.fillStyle = "rgba(255, 238, 77, 0.9)";
      ctx.arc(px, py, 1 + Math.random() * 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawPortalLabel(polygon, filterName, w);
}

function portalPolygon(points) {
  if (points.length !== 4) return [];

  const sorted = [...points].sort((a, b) => a.y - b.y);
  const top = sorted.slice(0, 2).sort((a, b) => a.x - b.x);
  const bottom = sorted.slice(2).sort((a, b) => a.x - b.x);

  // Same ordering as the Python implementation: top-left, top-right,
  // bottom-right, bottom-left.
  return [top[0], top[1], bottom[1], bottom[0]];
}

function drawPortalLabel(polygon, filterName, w) {
  const anchor = polygon.reduce((best, p) => (p.y < best.y ? p : best), polygon[0]);
  const label = `PORTAL: ${filterName}`;

  ctx.font = "600 15px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "bottom";
  const textWidth = ctx.measureText(label).width;
  const x = Math.min(Math.max(8, anchor.x), Math.max(8, w - textWidth - 8));
  const y = Math.max(24, anchor.y - 12);

  ctx.fillStyle = "rgba(5, 2, 12, 0.68)";
  ctx.fillRect(x - 7, y - 19, textWidth + 14, 24);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, x, y);
}

function drawFingerDot(point) {
  ctx.save();
  ctx.shadowColor = "#00ffff";
  ctx.shadowBlur = 12;
  ctx.fillStyle = "#00ffff";
  ctx.beginPath();
  ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/* ------------------------------------------------------------------ */
/* Filters                                                             */
/* ------------------------------------------------------------------ */

function renderFilteredRoi(x, y, w, h, filterName) {
  roiCanvas.width = w;
  roiCanvas.height = h;
  roiCtx.clearRect(0, 0, w, h);
  roiCtx.drawImage(canvas, x, y, w, h, 0, 0, w, h);

  switch (filterName) {
    case "MONO":
      applyMono();
      break;
    case "DUAL-TONE":
      applyDualTone();
      break;
    case "PIXELATE":
      applyPixelate(w, h);
      break;
    case "INVERT":
      applyInvert();
      break;
    case "SEPIA":
      applySepia();
      break;
    case "BLUR":
      applyBlur(w, h);
      break;
    case "THERMAL":
      applyThermal();
      break;
    case "SKETCH":
      applySketch(w, h);
      break;
    case "GLITCH":
      applyGlitch(w, h);
      break;
    case "NEON":
      applyNeon(w, h);
      break;
    case "GALAXY":
      applyGalaxy(x, y, w, h);
      break;
  }

  return roiCanvas;
}

function getRoiImage() {
  return roiCtx.getImageData(0, 0, roiCanvas.width, roiCanvas.height);
}

function putRoiImage(image) {
  roiCtx.putImageData(image, 0, 0);
}

function applyMono() {
  const image = getRoiImage();
  const d = image.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = (d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114 + 0.5) | 0;
    d[i] = d[i + 1] = d[i + 2] = v;
  }
  putRoiImage(image);
}

function applyInvert() {
  const image = getRoiImage();
  const d = image.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = 255 - d[i];
    d[i + 1] = 255 - d[i + 1];
    d[i + 2] = 255 - d[i + 2];
  }
  putRoiImage(image);
}

function applySepia() {
  const image = getRoiImage();
  const d = image.data;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i];
    const g = d[i + 1];
    const b = d[i + 2];
    d[i] = Math.min(255, 0.393 * r + 0.769 * g + 0.189 * b);
    d[i + 1] = Math.min(255, 0.349 * r + 0.686 * g + 0.168 * b);
    d[i + 2] = Math.min(255, 0.272 * r + 0.534 * g + 0.131 * b);
  }
  putRoiImage(image);
}

function applyDualTone() {
  const image = getRoiImage();
  const d = image.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
    if (v > 127) {
      d[i] = 255;
      d[i + 1] = 165;
      d[i + 2] = 0; // orange
    } else {
      d[i] = 255;
      d[i + 1] = 20;
      d[i + 2] = 147; // pink
    }
  }
  putRoiImage(image);
}

function applyThermal() {
  const image = getRoiImage();
  const d = image.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = (d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114) | 0;
    d[i] = JET.r[v];
    d[i + 1] = JET.g[v];
    d[i + 2] = JET.b[v];
  }
  putRoiImage(image);
}

function applyPixelate(w, h) {
  const sw = Math.max(1, Math.floor(w / 10));
  const sh = Math.max(1, Math.floor(h / 10));

  tmpCanvas.width = sw;
  tmpCanvas.height = sh;
  tmpCtx.imageSmoothingEnabled = true;
  tmpCtx.clearRect(0, 0, sw, sh);
  tmpCtx.drawImage(roiCanvas, 0, 0, w, h, 0, 0, sw, sh);

  roiCtx.clearRect(0, 0, w, h);
  roiCtx.imageSmoothingEnabled = false;
  roiCtx.drawImage(tmpCanvas, 0, 0, sw, sh, 0, 0, w, h);
  roiCtx.imageSmoothingEnabled = true;
}

function applyBlur(w, h) {
  encCanvas(tmpCanvas, tmpCtx, w, h);
  tmpCtx.filter = `blur(${Math.max(5, Math.round(Math.min(w, h) * 0.035))}px)`;
  tmpCtx.drawImage(roiCanvas, 0, 0);
  tmpCtx.filter = "none";

  roiCtx.clearRect(0, 0, w, h);
  roiCtx.drawImage(tmpCanvas, 0, 0);
}

function applySketch(w, h) {
  const image = getRoiImage();
  const d = image.data;
  const gray = new Uint8ClampedArray(w * h);

  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    gray[p] = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
  }

  encCanvas(tmpCanvas, tmpCtx, w, h);
  tmpCtx.filter = `grayscale(1) invert(1) blur(${Math.max(3, Math.round(Math.min(w, h) * 0.05))}px)`;
  tmpCtx.drawImage(roiCanvas, 0, 0);
  tmpCtx.filter = "none";
  const blurred = tmpCtx.getImageData(0, 0, w, h).data;

  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    const denom = 255 - blurred[i];
    const v = denom <= 0 ? 0 : Math.min(255, Math.round((gray[p] * 256) / denom));
    d[i] = d[i + 1] = d[i + 2] = v;
  }
  putRoiImage(image);
}

function applyGlitch(w, h) {
  const image = getRoiImage();
  const d = image.data;
  const src = d.slice();
  const shift = Math.max(5, Math.floor(w / 20));

  if (shift < w) {
    for (let row = 0; row < h; row++) {
      const offset = row * w * 4;
      for (let x = 0; x < w; x++) {
        const i = offset + x * 4;
        if (x + shift < w) d[i] = src[i + shift * 4]; // red channel shift
        if (x - shift >= 0) d[i + 2] = src[i - shift * 4 + 2]; // blue shift
      }
    }
  }
  putRoiImage(image);
}

function applyNeon(w, h) {
  const image = getRoiImage();
  const d = image.data;
  const gray = new Float32Array(w * h);

  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    gray[p] = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
  }

  const edges = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        -gray[i - w - 1] - 2 * gray[i - 1] - gray[i + w - 1] +
        gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1];
      const gy =
        -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1] +
        gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
      if (Math.abs(gx) + Math.abs(gy) > 180) edges[i] = 1;
    }
  }

  for (let i = 0; i < d.length; i += 4) {
    d[i] = 0;
    d[i + 1] = 0;
    d[i + 2] = 0;
  }

  // 3x3 dilation, cyan edges.
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      if (!edges[y * w + x]) continue;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const i = ((y + dy) * w + (x + dx)) * 4;
          d[i] = 0;
          d[i + 1] = 255;
          d[i + 2] = 255;
        }
      }
    }
  }
  putRoiImage(image);
}

function applyGalaxy(x, y, w, h) {
  if (!personMask || !galaxy) return; // same fallback as the Python app

  const image = getRoiImage();
  const d = image.data;
  const bg = galaxy.ctx.getImageData(x, y, w, h).data;
  const { data: mask, width: mw, height: mh } = personMask;
  const cw = canvas.width;
  const ch = canvas.height;

  for (let row = 0; row < h; row++) {
    const cy = y + row;
    const my = mh > 1 ? clamp(Math.round((cy / (ch - 1)) * (mh - 1)), 0, mh - 1) : 0;

    for (let col = 0; col < w; col++) {
      const cx = x + col;
      // The mask is in raw camera space; the canvas is mirrored.
      const mx = mw > 1 ? clamp(Math.round((1 - cx / (cw - 1)) * (mw - 1)), 0, mw - 1) : 0;

      if (mask[my * mw + mx] === 0) {
        const i = (row * w + col) * 4;
        d[i] = bg[i];
        d[i + 1] = bg[i + 1];
        d[i + 2] = bg[i + 2];
        d[i + 3] = 255;
      }
    }
  }
  putRoiImage(image);
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function buildGalaxy(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d");

  g.fillStyle = "rgb(40, 10, 30)"; // space purple (BGR 30,10,40 in Python)
  g.fillRect(0, 0, w, h);

  for (let i = 0; i < 800; i++) {
    g.fillStyle = `rgba(255, 255, 255, ${0.45 + Math.random() * 0.55})`;
    g.fillRect((Math.random() * w) | 0, (Math.random() * h) | 0, 1, 1);
  }

  for (let i = 0; i < 100; i++) {
    const r = 255;
    const green = 100 + Math.random() * 155;
    const blue = 150 + Math.random() * 105;
    g.fillStyle = `rgba(${r}, ${green | 0}, ${blue | 0}, ${0.5 + Math.random() * 0.5})`;
    g.beginPath();
    g.arc(Math.random() * w, Math.random() * h, 2 + Math.random() * 4, 0, Math.PI * 2);
    g.fill();
  }

  galaxy = { canvas: c, ctx: g };
}

function encCanvas(targetCanvas, targetCtx, w, h) {
  targetCanvas.width = w;
  targetCanvas.height = h;
  targetCtx.clearRect(0, 0, w, h);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function buildJetLut() {
  const r = new Uint8Array(256);
  const g = new Uint8Array(256);
  const b = new Uint8Array(256);

  for (let i = 0; i < 256; i++) {
    const v = i / 255;
    r[i] = Math.round(clamp(1.5 - Math.abs(4 * v - 3), 0, 1) * 255);
    g[i] = Math.round(clamp(1.5 - Math.abs(4 * v - 2), 0, 1) * 255);
    b[i] = Math.round(clamp(1.5 - Math.abs(4 * v - 1), 0, 1) * 255);
  }

  return { r, g, b };
}
