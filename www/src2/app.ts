import { css, html, LitElement } from "lit";
import { customElement, query, state } from "lit/decorators.js";
import * as tf from "@tensorflow/tfjs";
import { Point, TrackedObject, Tracker, TrackerOptions } from "./norfair";
import { HNSW } from "./hnsw/main";

interface Step {
  type: string;
  in_dim: number;
  out_dim: number;
  in_dtype: "float32" | "uint8";
  out_dtype: "float32" | "uint8";
}

interface LinearTransform extends Step {
  type: "LinearTransform";
  params: {
    A: number[][];
    b: number[];
  };
}

interface ScalarQuantizer extends Step {
  type: "ScalarQuantizer";
  params: {
    vmin: number[];
    vdiff: number[];
  };
  mode: "QT_8bit";
}

interface Meta {
  model?: string;
  chain: LinearTransform[];
  quantize: ScalarQuantizer;
  ids: string[];
}

// usearch naming:
// * connectivity = hnsw_connectivity,  # Number of Graph connections per layer of HNSW. Original paper calls it "M". Can't be changed after construction.
// * expansion_add = hnsw_expansion_add,  # Search depth when inserting new vectors. Original paper calls it "efConstruction". Can be changed afterwards.
// * expansion_search = hnsw_expansion_search,  # Search depth when querying nearest neighbors. Original paper calls it "ef". Can be changed afterwards.

const cosineSimilarity = (a: number[], b: number[]) => {
  let A = 0;
  let B = 0;
  let AB = 0;
  for (let i = 0; i < a.length; i++) {
    A += a[i] * a[i];
    B += b[i] * b[i];
    AB += a[i] * b[i];
  }
  if (A === 0 || B === 0) {
    return 1;
  }
  return AB / (Math.sqrt(A) * Math.sqrt(B));
};

const cosineSimilarityPreNorm = (a: number[], b: number[]) => {
  let AB = 0;
  for (let i = 0; i < a.length; i++) {
    AB += a[i] * b[i];
  }
  return AB;
};

const makeLinearForward = (params: LinearTransform) => {
  // ORIGINAL PYTHON:
  // | output = np.zeros(reduce_dim, dtype=v.dtype)
  // | for i in range(reduce_dim):
  // |     total = b_vec[i]
  // |     for j in range(D):
  // |         total += A_mat[i, j] * v[j]
  // |     output[i] = total
  // | v = output
  // params
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const b = params.params.b;
  const A = params.params.A;

  return (vec: number[]): number[] => {
    // compute
    const output = new Array(outDim);
    for (let i = 0; i < outDim; i++) {
      let total = b[i];
      for (let j = 0; j < inDim; j++) {
        total += A[i][j] * vec[j];
      }
      output[i] = total;
    }
    return output;
  };
};

const makeEncode = (params: ScalarQuantizer) => {
  // ORIGINAL PYTHON:
  // | # https://github.com/facebookresearch/faiss/blob/d4fa401656fa413728f3c93bae4e34fb81803d54/faiss/impl/ScalarQuantizer.cpp#L381
  // | out = np.zeros((len(v),), dtype=np.uint8)
  // | for i in range(len(v)):
  // |     vd = vdiff[i]
  // |     vm = vmin[i]
  // |     xi = 0
  // |     if vd != 0:
  // |         xi = (v[i] - vm) / vd
  // |         if xi < 0:
  // |             xi = 0
  // |         if xi > 1.0:
  // |             xi = 1.0
  // |     out[i] = int(xi * 255)
  // | v = out
  // OR
  // | def _manual_encode(x):
  // |     x = (x - vmin) / vdiff
  // |     x = np.clip(x * 255, 0, 255).astype(np.uint8)
  // |     return x

  // params
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const vmin = params.params.vmin;
  const vdiff = params.params.vdiff;

  if (inDim !== outDim) {
    throw new Error("in_dim != out_dim");
  }

  return (vec: number[]): Uint8Array => {
    // compute
    const out = new Uint8Array(outDim);
    for (let i = 0; i < inDim; i++) {
      const vd = vdiff[i];
      const vm = vmin[i];
      let xi = 0;
      if (vd !== 0) {
        xi = (vec[i] - vm) / vd;
        if (xi < 0) {
          xi = 0;
        }
        if (xi > 1.0) {
          xi = 1.0;
        }
      }
      out[i] = Math.round(xi * 255);
    }
    return out;
  };
};

const makeDecode = (params: ScalarQuantizer) => {
  // ORIGINAL PYTHON:
  // | def _manual_decode(x):
  // |     x = (x + 0.5) / 255
  // |     x = vmin + x * vdiff
  // |     return x

  // params
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const vmin = params.params.vmin;
  const vdiff = params.params.vdiff;

  if (inDim !== outDim) {
    throw new Error("in_dim != out_dim");
  }

  return (vec: Uint8Array): number[] => {
    const out = new Array(outDim);
    for (let i = 0; i < inDim; i++) {
      out[i] = (vmin[i] + (vec[i] + 0.5) / 255) * vdiff[i];
    }
    return out;
  };
};

const makeDecodeSimilarityPreNorm = (params: ScalarQuantizer) => {
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const vmin = params.params.vmin;
  const vdiff = params.params.vdiff;

  if (inDim !== outDim) {
    throw new Error("in_dim != out_dim");
  }

  return (a: Uint8Array, b: Uint8Array) => {
    let AB = 0;
    for (let i = 0; i < inDim; i++) {
      const A = (vmin[i] + (a[i] + 0.5) / 255) * vdiff[i];
      const B = (vmin[i] + (b[i] + 0.5) / 255) * vdiff[i];
      AB += A * B;
    }
    return AB;
  };
};

const makeDecodeSimilarity = (params: ScalarQuantizer) => {
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const vmin = params.params.vmin;
  const vdiff = params.params.vdiff;

  if (inDim !== outDim) {
    throw new Error("in_dim != out_dim");
  }

  return (a: Uint8Array, b: Uint8Array) => {
    let AB = 0;
    let AA = 0;
    let BB = 0;
    for (let i = 0; i < inDim; i++) {
      const A = (vmin[i] + (a[i] + 0.5) / 255) * vdiff[i];
      const B = (vmin[i] + (b[i] + 0.5) / 255) * vdiff[i];
      AB += A * B;
      AA += A * A;
      BB += B * B;
    }
    return AB / (Math.sqrt(AA) * Math.sqrt(BB));
  };
};

let index: HNSW;
let indexMeta: Meta;
let indexProcessVec: (vec: number[]) => number[];
let indexLoadState = "N/A";

async function loadIndex() {
  if (index || indexMeta) {
    console.warn("Index already loaded.");
    return;
  }

  console.log("Loading index...");
  indexLoadState = "Fetching Data...";

  const metaFile = "/index/index_meta.json";
  const vecFile = "/index/index_vecs.bin";

  // load metaFile
  console.log("load meta...");
  const metaResponse = await fetch(metaFile);
  const meta: Meta = await metaResponse.json();

  // load index
  console.log("load vecs...");
  const vecsResponse = await fetch(vecFile);
  const vecsBuffer = await vecsResponse.arrayBuffer();
  const vecs = new Uint8Array(vecsBuffer);

  const getVec = (i: number) => {
    const byteStart = i * vecQuantBytes;
    const byteEnd = byteStart + vecQuantBytes;
    const vec = vecs.slice(byteStart, byteEnd);
    return vec;
  };

  // check
  const numVecs = meta.ids.length;
  const vecQuantSize = meta.quantize.out_dim;
  const bytesPerNumQuant = { float32: 4, uint8: 1 }[meta.quantize.out_dtype];
  const vecQuantBytes = vecQuantSize * bytesPerNumQuant;
  const totalBytes = numVecs * vecQuantBytes;

  if (vecs.length !== totalBytes) {
    throw new Error(
      `Index vecs size mismatch: ${vecs.length} != ${totalBytes}`,
    );
  }
  if (meta.quantize.out_dtype !== "uint8") {
    throw new Error(
      `Index quantize out_dtype not yet supported: ${meta.quantize.out_dtype}, expected uint8`,
    );
  }
  if (meta.chain.length !== 1) {
    throw new Error(
      `Index chain length not yet supported: ${meta.chain.length}, expected 1`,
    );
  }

  const decode = makeDecode(meta.quantize);
  const decodeSimilarity = makeDecodeSimilarity(meta.quantize);
  const decodeSimilarityPreNorm = makeDecodeSimilarityPreNorm(meta.quantize);

  const quantizedCosineSimilarity = (a: number[], b: number[]) => {
    const aIsRef = a.length == 1;
    const bIsRef = b.length == 1;
    if (aIsRef && bIsRef) {
      return decodeSimilarity(getVec(a[0]), getVec(b[0]));
    }
    const _a = aIsRef ? decode(getVec(a[0])) : a;
    const _b = bIsRef ? decode(getVec(b[0])) : b;
    return cosineSimilarity(_a, _b);
  };

  const quantizedCosineSimilarityPreNormalized = (a: number[], b: number[]) => {
    const aIsRef = a.length == 1;
    const bIsRef = b.length == 1;
    if (aIsRef && bIsRef) {
      return decodeSimilarityPreNorm(getVec(a[0]), getVec(b[0]));
    }
    const _a = aIsRef ? decode(getVec(a[0])) : a;
    const _b = bIsRef ? decode(getVec(b[0])) : b;
    return cosineSimilarityPreNorm(_a, _b);
  };

  // create index
  indexLoadState = "Creating Index...";
  const idx = new HNSW(16, 100, vecQuantSize, quantizedCosineSimilarity);

  // add vectors to index
  // THIS IS VERY SLOW... probably need a loading indicator...
  console.log("add vecs...");

  const time = performance.now();
  for (let i = 0; i < meta.ids.length; i++) {
    await idx.addPoint(meta.ids[i], [i]);
  }
  const time2 = performance.now();

  console.log("Loaded index in", time2 - time, "ms");
  meta.ids = null;
  indexMeta = meta;
  index = idx;
  indexProcessVec = makeLinearForward(meta.chain[0]);
  indexLoadState = "Success";
}

// ===========================================================
// Configuration Interfaces
// ===========================================================

interface YoloConfig {
  modelUrl: string;
  inputWidth: number;
  inputHeight: number;
  confidenceThreshold: number;
  detBoxCoeffIndex: number; // Index where mask coefficients start
  protoMaskSize: number;
  maskCoeffCount: number;
}

interface EmbeddingConfig {
  modelUrl: string;
  inputWidth: number;
  inputHeight: number;
  cropPaddingFactor: number;
}

interface DrawingConfig {
  maskThreshold: number;
  maskCropPaddingFactor: number; // Padding used during mask generation affects drawing calculation
  maskDrawAlpha: number;
  colors: number[][];
}

interface TrackerConfig extends TrackerOptions {} // Use Norfair's options directly

// ===========================================================
// Type Definitions (Shared Interfaces)
// ===========================================================

interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
} // Video coordinates
interface BBoxCanvas {
  x: number;
  y: number;
  w: number;
  h: number;
} // Canvas coordinates
interface VideoDims {
  w: number;
  h: number;
}

interface RawDetection {
  bboxModel: number[]; // [x1, y1, x2, y2] in model input coords
  confidence: number;
  classId: number;
  maskCoeffs: tf.Tensor; // Shape [1, 1, num_coeffs], owned by caller of YoloProcessor.processFrame
}

interface DrawInfo {
  id: number;
  color: number[];
  bboxCanvas: BBoxCanvas;
  label: string;
  mask?: {
    tensor: tf.Tensor2D; // Binary mask tensor
    videoBox: BBox; // Video coordinate box the mask corresponds to (reflecting padding used for generation)
  };
}

interface ObjectEmbeddingInfo {
  id: number;
  hasBeenEmbedded: boolean;
  lastEmbeddingTime: number | null;
  creationTime: number;
  embedding: tf.Tensor | null; // Stores the embedding tensor (ownership with this object)
  lastKnownBboxVideo: BBox | null;
  matchId: string | null;
  lastMatchTime: number | null;
}

// ===========================================================
// Constants -> Configuration Objects
// ===========================================================

const YOLO_CONFIG: YoloConfig = {
  modelUrl: "/assets/models/yolov11s_seg__dk964hap__web_model/model.json",
  inputWidth: 640,
  inputHeight: 640,
  confidenceThreshold: 0.5,
  detBoxCoeffIndex: 6,
  protoMaskSize: 160,
  maskCoeffCount: 32,
};

const EMBEDDING_CONFIG: EmbeddingConfig = {
  modelUrl:
    "/assets/models/convnextv2_convlinear__aivb8jvk-47500__encoder__web_model/model.json",
  inputWidth: 128,
  inputHeight: 192,
  cropPaddingFactor: 0.1,
};

const DRAWING_CONFIG: DrawingConfig = {
  maskThreshold: 0.5,
  maskCropPaddingFactor: 0.0, // Keep at 0.0 for testing alignment first
  maskDrawAlpha: 0.5,
  colors: [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [255, 165, 0],
    [255, 192, 203],
    [75, 0, 130],
  ],
};

const TRACKER_CONFIG: TrackerConfig = {
  distanceThreshold: 100,
  hitInertiaMin: 0,
  hitInertiaMax: 15,
  initDelay: 2,
};

const EMBEDDING_LOOP_INTERVAL_MS = 250;

// ===========================================================
// Functional Mask Generator
// ===========================================================

/**
 * Generates a binary mask tensor for a single detection using a functional approach.
 * Disposes intermediate tensors it creates via tf.tidy.
 * Does NOT dispose input tensors (protoTensor, maskCoeffs).
 * Returns null if generation fails or inputs are invalid.
 */
async function generateMask(
  protoTensor: tf.Tensor,
  maskCoeffs: tf.Tensor,
  targetVideoBox: BBox, // The UNPADDED video box of the detection
  videoDims: VideoDims,
  config: {
    protoSize: number;
    coeffCount: number;
    paddingFactor: number;
    threshold: number;
  },
): Promise<tf.Tensor2D | null> {
  // Input validation
  if (
    !protoTensor ||
    protoTensor.isDisposed ||
    !maskCoeffs ||
    maskCoeffs.isDisposed
  ) {
    console.warn("generateMask received disposed tensor.");
    return null;
  }
  if (!targetVideoBox || !videoDims || videoDims.w <= 0 || videoDims.h <= 0) {
    console.warn("generateMask received invalid bbox or video dimensions.");
    return null;
  }

  let finalMask: tf.Tensor2D | null = null;
  try {
    finalMask = tf.tidy(() => {
      // 1. Combine protos and coefficients
      const protosReshaped = protoTensor
        .squeeze(0)
        .reshape([config.protoSize * config.protoSize, config.coeffCount]);
      const coeffsReshaped = maskCoeffs.reshape([config.coeffCount, 1]);
      const maskProto = tf.matMul(protosReshaped, coeffsReshaped);
      const maskReshaped = maskProto.reshape([
        config.protoSize,
        config.protoSize,
      ]);
      const maskActivated = tf.sigmoid(maskReshaped);

      // 2. Calculate Crop Box based on TARGET video box and padding
      const boxWidth = targetVideoBox.x2 - targetVideoBox.x1;
      const boxHeight = targetVideoBox.y2 - targetVideoBox.y1;
      const padX = boxWidth * config.paddingFactor;
      const padY = boxHeight * config.paddingFactor;
      const cropX1 = Math.max(0, targetVideoBox.x1 - padX);
      const cropY1 = Math.max(0, targetVideoBox.y1 - padY);
      const cropX2 = Math.min(videoDims.w, targetVideoBox.x2 + padX);
      const cropY2 = Math.min(videoDims.h, targetVideoBox.y2 + padY);
      // Target dimensions for the output mask tensor
      const targetCropW = Math.ceil(cropX2 - cropX1);
      const targetCropH = Math.ceil(cropY2 - cropY1);

      if (targetCropW <= 0 || targetCropH <= 0) return null;

      // Normalized coordinates of the area to crop *from the prototype mask*
      const normalizedPaddedBbox = [
        [
          cropY1 / videoDims.h,
          cropX1 / videoDims.w,
          cropY2 / videoDims.h,
          cropX2 / videoDims.w,
        ],
      ];

      // 3. Crop and Resize the activated prototype mask
      const maskExpanded = maskActivated.expandDims(0).expandDims(-1);
      const maskCroppedResized = tf.image.cropAndResize(
        maskExpanded,
        normalizedPaddedBbox,
        [0],
        [targetCropH, targetCropW],
        "bilinear",
      );

      // 4. Threshold to get binary mask
      const mask = maskCroppedResized
        .squeeze([0, 3])
        .greater(config.threshold)
        .cast("float32");
      return mask; // Return the final tensor from tidy
    });
  } catch (error) {
    console.error("Error generating mask:", error);
    finalMask?.dispose();
    return null;
  }
  // Ownership of finalMask (if not null) is transferred to the caller.
  return finalMask as tf.Tensor2D | null;
}

// ===========================================================
// Service Class Implementations
// ===========================================================

// --- 1. VideoSourceManager --- (No changes)
class VideoSourceManager {
  private videoElement: HTMLVideoElement | null = null;
  private currentStream: MediaStream | null = null;
  private streamError: string | null = null;
  constructor() {}
  async startStream(
    element: HTMLVideoElement,
    constraints?: MediaStreamConstraints,
  ): Promise<void> {
    /* ... */
    this.videoElement = element;
    this.streamError = null;
    if (this.currentStream) {
      this.stopStream();
    }
    try {
      const defaultConstraints = {
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      };
      this.currentStream = await navigator.mediaDevices.getUserMedia(
        constraints || defaultConstraints,
      );
      if (this.videoElement) {
        this.videoElement.srcObject = this.currentStream;
        await this.videoElement.play(); /* console.log("Video stream started."); */
      } else {
        throw new Error("Video element not set.");
      }
    } catch (error: any) {
      console.error("Camera failed:", error);
      this.streamError = `Camera failed: ${error.message}`;
      this.stopStream();
      throw error;
    }
  }
  stopStream(): void {
    /* ... */
    if (this.currentStream) {
      this.currentStream.getTracks().forEach((track) => track.stop());
    }
    this.currentStream = null;
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.srcObject = null;
    }
  }
  getVideoElement(): HTMLVideoElement | null {
    return this.videoElement;
  }
  getVideoDimensions(): VideoDims | null {
    if (this.videoElement?.videoWidth) {
      return {
        w: this.videoElement.videoWidth,
        h: this.videoElement.videoHeight,
      };
    }
    return null;
  }
  isReady(): boolean {
    return !!(
      this.videoElement &&
      this.currentStream &&
      this.videoElement.readyState >= 4 &&
      !this.videoElement.paused
    );
  } // Use readyState 4 (HAVE_ENOUGH_DATA)
  getError(): string | null {
    return this.streamError;
  }
}

// --- 2. ModelLoader Utility --- (No changes)
class ModelLoader {
  static async loadModel(url: string): Promise<tf.GraphModel> {
    /* ... */
    try {
      const model = await tf.loadGraphModel(url);
      console.log(`Model loaded: ${url}`);
      return model;
    } catch (error: any) {
      console.error(`Failed load: ${url}:`, error);
      throw new Error(`Model load failed (${url}): ${error.message || error}`);
    }
  }
}

// --- 3. YoloProcessor --- (No changes from previous refactor)
class YoloProcessor {
  constructor(
    private yoloModel: tf.GraphModel,
    private config: YoloConfig,
  ) {}
  async processFrame(
    frameTensor: tf.Tensor,
  ): Promise<{ detections: RawDetection[]; protoTensor: tf.Tensor | null }> {
    /* ... */
    let output: tf.Tensor[] | null = null;
    const detections: RawDetection[] = [];
    let protoTensor: tf.Tensor | null = null;
    const tensorsCreated: tf.Tensor[] = [];
    try {
      const inputTensor = tf.tidy(() => {
        const r = tf.image.resizeBilinear(frameTensor, [
          this.config.inputHeight,
          this.config.inputWidth,
        ]);
        return r.div(255.0).expandDims(0).cast("float32");
      });
      tensorsCreated.push(inputTensor);
      output = (await this.yoloModel.executeAsync(inputTensor)) as tf.Tensor[];
      if (!output || output.length < 3) {
        console.warn("YOLO output short:", output?.length);
        inputTensor.dispose();
        return { detections: [], protoTensor: null };
      }
      tensorsCreated.push(...output);
      const detectionsTensor = output[0];
      protoTensor = output[2];
      if (
        protoTensor.shape.length !== 4 ||
        protoTensor.shape[1] !== this.config.protoMaskSize ||
        protoTensor.shape[3] !== this.config.maskCoeffCount
      ) {
        throw new Error("Incorrect proto mask shape.");
      }
      const detectionsBatch = (await detectionsTensor.data()) as Float32Array;
      const numDets = detectionsTensor.shape[1];
      const detDataLength = detectionsTensor.shape[2];
      for (let i = 0; i < numDets; i++) {
        const offset = i * detDataLength;
        const confidence = detectionsBatch[offset + 4];
        if (confidence >= this.config.confidenceThreshold) {
          const maskCoeffs = tf.slice(
            detectionsTensor,
            [0, i, this.config.detBoxCoeffIndex],
            [1, 1, this.config.maskCoeffCount],
          );
          detections.push({
            bboxModel: Array.from(detectionsBatch.slice(offset, offset + 4)),
            confidence: confidence,
            classId: Math.round(detectionsBatch[offset + 5]),
            maskCoeffs: maskCoeffs,
          });
        }
      }
      detectionsTensor.dispose();
      inputTensor.dispose();
      if (output[1] && !output[1].isDisposed) output[1].dispose();
    } catch (error) {
      console.error("Error YOLO:", error);
      tf.dispose(tensorsCreated);
      tf.dispose(detections.map((d) => d.maskCoeffs));
      return { detections: [], protoTensor: null };
    }
    return { detections, protoTensor };
  }
}

// --- 4. MaskGenerator FUNCTION --- (Defined above)

// --- 5. ObjectTracker (Norfair Wrapper) --- (No changes)
class ObjectTracker {
  private tracker: Tracker;
  constructor(config: TrackerConfig) {
    this.tracker = new Tracker(config);
    console.log("Norfair tracker init");
  }
  update(detections: { point: Point }[]): number[] {
    return this.tracker.update(detections.map((d) => d.point));
  }
  getTrackedObjects(): TrackedObject[] {
    return this.tracker.trackedObjects;
  }
}

// --- 6. EmbeddingProcessor --- (No changes from previous refactor)
class EmbeddingProcessor {
  constructor(
    private embedModel: tf.GraphModel,
    private config: EmbeddingConfig,
  ) {}
  async createEmbedding(
    videoElement: HTMLVideoElement,
    targetVideoBox: BBox,
  ): Promise<tf.Tensor | null> {
    /* ... */
    if (!videoElement.videoWidth || !this.embedModel) return null;
    let embedding: tf.Tensor | null = null;
    let clonedEmbedding: tf.Tensor | null = null;
    try {
      const vW = videoElement.videoWidth;
      const vH = videoElement.videoHeight;
      const bW = targetVideoBox.x2 - targetVideoBox.x1;
      const bH = targetVideoBox.y2 - targetVideoBox.y1;
      const pX = bW * this.config.cropPaddingFactor;
      const pY = bH * this.config.cropPaddingFactor;
      const cX1 = Math.max(0, Math.floor(targetVideoBox.x1 - pX));
      const cY1 = Math.max(0, Math.floor(targetVideoBox.y1 - pY));
      const cX2 = Math.min(vW, Math.ceil(targetVideoBox.x2 + pX));
      const cY2 = Math.min(vH, Math.ceil(targetVideoBox.y2 + pY));
      if (cX2 <= cX1 || cY2 <= cY1) {
        console.warn("Invalid crop embed.");
        return null;
      }
      const boxes = [[cY1 / vH, cX1 / vW, cY2 / vH, cX2 / vW]];
      const cropSize: [number, number] = [
        this.config.inputHeight,
        this.config.inputWidth,
      ];
      embedding = await tf.tidy(() => {
        const f = tf.browser.fromPixels(videoElement);
        const c = tf.image.cropAndResize(
          f.expandDims(0).toFloat(),
          boxes,
          [0],
          cropSize,
          "bilinear",
        );
        return this.embedModel.execute(c.div(255.0)) as tf.Tensor;
      });
      if (embedding?.shape?.length > 0) {
        clonedEmbedding = embedding.clone();
      } else {
        console.warn("Embed invalid tensor.");
      }
    } catch (error) {
      console.error("Error embedding:", error);
    } finally {
      embedding?.dispose();
    }
    return clonedEmbedding;
  }
}

// --- 7. CanvasRenderer --- (No changes from previous refactor)
class CanvasRenderer {
  private visibleCtx: CanvasRenderingContext2D;
  private bufferCanvas: HTMLCanvasElement;
  private bufferCtx: CanvasRenderingContext2D;
  private displayWidth = 0;
  private displayHeight = 0;
  private videoWidth = 0;
  private videoHeight = 0;
  public scale = 1;
  public offsetX = 0;
  public offsetY = 0;
  constructor(
    private visibleCanvas: HTMLCanvasElement,
    private drawingConfig: DrawingConfig,
  ) {
    /* ... */
    const ctx = visibleCanvas.getContext("2d");
    if (!ctx) throw new Error("No visible ctx");
    this.visibleCtx = ctx;
    this.bufferCanvas = document.createElement("canvas");
    const bufCtx = this.bufferCanvas.getContext("2d");
    if (!bufCtx) throw new Error("No buffer ctx");
    this.bufferCtx = bufCtx;
  }
  updateDimensions(videoElement: HTMLVideoElement): boolean {
    /* ... */
    if (!videoElement.videoWidth) return false;
    const dW = videoElement.clientWidth;
    const dH = videoElement.clientHeight;
    const vW = videoElement.videoWidth;
    const vH = videoElement.videoHeight;
    let changed = false;
    if (this.displayWidth !== dW || this.displayHeight !== dH) {
      this.visibleCanvas.width = dW;
      this.visibleCanvas.height = dH;
      this.bufferCanvas.width = dW;
      this.bufferCanvas.height = dH;
      this.displayWidth = dW;
      this.displayHeight = dH;
      changed = true;
    }
    if (this.videoWidth !== vW || this.videoHeight !== vH || changed) {
      this.videoWidth = vW;
      this.videoHeight = vH;
      const sX = dW / vW;
      const sY = dH / vH;
      this.scale = Math.min(sX, sY);
      this.offsetX = (dW - vW * this.scale) / 2;
      this.offsetY = (dH - vH * this.scale) / 2;
      changed = true;
    }
    return changed;
  }
  convertVideoBoxToCanvas(videoBox: BBox): BBoxCanvas {
    /* ... */ return {
      x: videoBox.x1 * this.scale + this.offsetX,
      y: videoBox.y1 * this.scale + this.offsetY,
      w: (videoBox.x2 - videoBox.x1) * this.scale,
      h: (videoBox.y2 - videoBox.y1) * this.scale,
    };
  }
  clearAll(): void {
    /* ... */ this.visibleCtx.clearRect(
      0,
      0,
      this.displayWidth,
      this.displayHeight,
    );
    this.bufferCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
  }
  async drawFrame(objectsToDraw: DrawInfo[]): Promise<void> {
    /* ... */
    this.bufferCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
    const maskPromises: Promise<void>[] = [];
    for (const obj of objectsToDraw) {
      this.drawBoundingBox(
        this.bufferCtx,
        obj.bboxCanvas,
        obj.color,
        obj.label,
      );
      if (obj.mask?.tensor && !obj.mask.tensor.isDisposed) {
        maskPromises.push(
          this.drawMask(
            this.bufferCtx,
            obj.mask.tensor,
            obj.color,
            obj.mask.videoBox,
          ),
        );
      }
    }
    await Promise.all(maskPromises);
    this.visibleCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
    this.visibleCtx.drawImage(this.bufferCanvas, 0, 0);
  }
  private drawBoundingBox(
    ctx: CanvasRenderingContext2D,
    bbox: BBoxCanvas,
    color: number[],
    label: string,
  ): void {
    /* ... */
    ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox.x, bbox.y, bbox.w, bbox.h);
    ctx.font = "12px sans-serif";
    ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.8)`;
    const m = ctx.measureText(label);
    const tW = m.width;
    const tH = 12;
    const p = 2;
    let tY = bbox.y - p - 1;
    let bY = tY - tH - p;
    if (bY < 0) {
      tY = bbox.y + bbox.h + tH + p;
      bY = bbox.y + bbox.h + p;
    }
    ctx.fillRect(bbox.x - 1, bY, tW + p * 2, tH + p * 2);
    ctx.fillStyle = `white`;
    ctx.fillText(label, bbox.x + p - 1, tY);
  }
  private async drawMask(
    drawCtx: CanvasRenderingContext2D,
    maskTensor: tf.Tensor2D,
    color: number[],
    videoBox: BBox,
  ): Promise<void> {
    /* ... */
    let tempMaskCanvas: HTMLCanvasElement | null = null;
    const tensorsToDispose: tf.Tensor[] = [];
    try {
      const [mH, mW] = maskTensor.shape;
      const bW = videoBox.x2 - videoBox.x1;
      const bH = videoBox.y2 - videoBox.y1;
      const pX = bW * this.drawingConfig.maskCropPaddingFactor;
      const pY = bH * this.drawingConfig.maskCropPaddingFactor;
      const mX1 = Math.max(0, videoBox.x1 - pX);
      const mY1 = Math.max(0, videoBox.y1 - pY);
      const vW = bW + 2 * pX;
      const vH = bH + 2 * pY; // Effective video size of mask area
      const colorTensor = tf.tensor1d(color.map((c) => c / 255.0));
      tensorsToDispose.push(colorTensor);
      const maskColored = tf.tidy(() =>
        maskTensor.expandDims(-1).mul(colorTensor.reshape([1, 1, 3])),
      );
      tensorsToDispose.push(maskColored);
      const alphaChannel = maskTensor.expandDims(-1);
      tensorsToDispose.push(alphaChannel);
      const maskRgba = tf.tidy(() =>
        tf.concat([maskColored, alphaChannel], -1),
      );
      tensorsToDispose.push(maskRgba);
      const maskPixelData = await tf.browser.toPixels(maskRgba);
      const imageData = new ImageData(maskPixelData, mW, mH);
      tempMaskCanvas = document.createElement("canvas");
      tempMaskCanvas.width = mW;
      tempMaskCanvas.height = mH;
      const tempCtx = tempMaskCanvas.getContext("2d");
      if (!tempCtx) throw new Error("No temp ctx");
      tempCtx.putImageData(imageData, 0, 0);
      const cX = mX1 * this.scale + this.offsetX;
      const cY = mY1 * this.scale + this.offsetY;
      const cW = vW * this.scale;
      const cH = vH * this.scale;
      drawCtx.save();
      drawCtx.globalAlpha = this.drawingConfig.maskDrawAlpha;
      if (cW > 0 && cH > 0) {
        drawCtx.drawImage(tempMaskCanvas, cX, cY, cW, cH);
      }
      drawCtx.restore();
    } catch (error) {
      console.error("Error drawing mask:", error);
    } finally {
      tf.dispose(tensorsToDispose);
    }
  }
}

// --- 8. TrackedObjectState --- (New State Manager)
class TrackedObjectState {
  private state = new Map<number, ObjectEmbeddingInfo>();

  updateTrackedObjects(
    trackedObjects: TrackedObject[],
    timestamp: number,
  ): void {
    const currentIds = new Set<number>();
    trackedObjects.forEach((obj) => {
      if (obj.id === -1) return;
      currentIds.add(obj.id);
      if (!this.state.has(obj.id)) {
        this.state.set(obj.id, {
          id: obj.id,
          hasBeenEmbedded: false,
          lastEmbeddingTime: null,
          creationTime: timestamp,
          embedding: null,
          lastKnownBboxVideo: null,
          matchId: null,
          lastMatchTime: null,
        });
      }
    });
    this.cleanupDeadObjects(currentIds); // Cleanup immediately after update
  }

  cleanupDeadObjects(activeIds: Set<number>): void {
    const deadIds: number[] = [];
    for (const id of this.state.keys()) {
      if (!activeIds.has(id)) {
        deadIds.push(id);
      }
    }
    deadIds.forEach((id) => {
      this.state.get(id)?.embedding?.dispose();
      this.state.delete(id);
    });
  }

  updateLastKnownBbox(id: number, bbox: BBox | null): void {
    const info = this.state.get(id);
    if (info) {
      info.lastKnownBboxVideo = bbox;
    }
  }

  selectObjectForEmbedding(): { id: number; bboxVideo: BBox } | null {
    const candidates = Array.from(this.state.values()).filter(
      (info) => info.lastKnownBboxVideo,
    );
    const unembedded = candidates
      .filter((i) => !i.hasBeenEmbedded)
      .sort((a, b) => a.creationTime - b.creationTime);
    if (unembedded.length) {
      return {
        id: unembedded[0].id,
        bboxVideo: unembedded[0].lastKnownBboxVideo!,
      };
    }
    const embedded = candidates
      .filter((i) => i.hasBeenEmbedded && i.lastEmbeddingTime)
      .sort((a, b) => a.lastEmbeddingTime! - b.lastEmbeddingTime!);
    if (embedded.length) {
      return { id: embedded[0].id, bboxVideo: embedded[0].lastKnownBboxVideo! };
    }
    const fallback = candidates.filter((i) => i.hasBeenEmbedded);
    if (fallback.length) {
      return { id: fallback[0].id, bboxVideo: fallback[0].lastKnownBboxVideo! };
    }
    return null;
  }

  get(id: number): ObjectEmbeddingInfo | null {
    return this.state.get(id) || null;
  }

  getEmbedding(id: number): tf.Tensor | null {
    return this.state.get(id)?.embedding || null;
  }

  updateEmbedding(id: number, embedding: tf.Tensor | null): void {
    const info = this.state.get(id);
    if (info) {
      info.embedding?.dispose();
      info.embedding = embedding;
      if (embedding) {
        info.hasBeenEmbedded = true;
        info.lastEmbeddingTime = Date.now();
      }
    } else {
      console.warn(`Update embed for non-existent ID ${id}`);
      embedding?.dispose();
    }
  }

  updateMatchingId(uid: string) {
    const info = this.state.get(parseInt(uid));
    if (info) {
      info.matchId = uid;
      info.lastMatchTime = Date.now();
    } else {
      console.warn(`Update match ID for non-existent ID ${uid}`);
    }
  }

  disposeAllEmbeddings(): void {
    this.state.forEach((info) => info.embedding?.dispose());
    this.state.clear();
  }
}

// ===========================================================
// Main LitElement Component (Orchestrator - V3.1)
// ===========================================================

@customElement("video-container")
class VideoContainer extends LitElement {
  static styles = css`
    /* ... (Styles remain the same) ... */
    :host {
      display: block;
      width: 100%;
      height: 100%;
      overflow: hidden;
      position: relative;
      background-color: #222;
    }
    video {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background-color: #000;
    }
    canvas#overlay-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 5;
    }
    .status {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background-color: rgba(0, 0, 0, 0.7);
      color: white;
      font-size: 0.9em;
      padding: 5px 10px;
      text-align: center;
      z-index: 10;
    }
  `;

  // --- Element Refs ---
  @query("#video") private videoElement!: HTMLVideoElement;
  @query("#overlay-canvas") private canvasElement!: HTMLCanvasElement;

  // --- State ---
  @state() private inferenceStatus: string = "Initializing...";
  @state() private modelsReady: boolean = false;
  @state() private streamReady: boolean = false;

  // --- Service Instances ---
  private videoSourceManager!: VideoSourceManager;
  private yoloProcessor!: YoloProcessor;
  private objectTracker!: ObjectTracker;
  private embeddingProcessor!: EmbeddingProcessor;
  private canvasRenderer!: CanvasRenderer;
  private trackedObjectState!: TrackedObjectState; // Use state manager
  private yoloModel: tf.GraphModel | null = null;
  private embedModel: tf.GraphModel | null = null;

  // --- Control Flow ---
  private inferenceLoopId: number | null = null;
  private embeddingLoopTimeoutId: number | null = null;
  private isInferencing: boolean = false;
  private isEmbedding: boolean = false;
  private resizeObserver!: ResizeObserver;

  // --- Lit Lifecycle ---

  connectedCallback() {
    super.connectedCallback();
    this.videoSourceManager = new VideoSourceManager();
    this.objectTracker = new ObjectTracker(TRACKER_CONFIG);
    this.trackedObjectState = new TrackedObjectState();
    tf.ready()
      .then(() => {
        this.loadModels();
      })
      .catch((tfError) => {
        this.handleFatalError("TF.js init failed", tfError);
      });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.stopLoops();
    this.videoSourceManager?.stopStream();
    this.resizeObserver?.disconnect();
    this.yoloModel?.dispose();
    this.embedModel?.dispose();
    this.trackedObjectState?.disposeAllEmbeddings();
    this.canvasRenderer?.clearAll();
    console.log("VideoContainer disconnected & cleaned up.");
  }

  protected firstUpdated() {
    if (!this.videoElement || !this.canvasElement) {
      this.handleFatalError("Elements missing.");
      return;
    }
    try {
      this.canvasRenderer = new CanvasRenderer(
        this.canvasElement,
        DRAWING_CONFIG,
      );
      this.resizeObserver = new ResizeObserver(() => {
        const vid = this.videoSourceManager?.getVideoElement();
        if (vid) this.canvasRenderer?.updateDimensions(vid);
      });
      this.resizeObserver.observe(this.videoElement);
      this.videoElement.addEventListener(
        "loadedmetadata",
        this.handleVideoMetadataLoaded,
      );
      this.videoElement.addEventListener("error", (e) =>
        this.handleFatalError("Video error.", e),
      );
      this.startVideoStream();
    } catch (error) {
      this.handleFatalError("firstUpdated error", error);
    }
  }

  // --- Orchestration ---

  private async loadModels() {
    /* ... (No changes needed from V3) ... */
    this.inferenceStatus = "Loading models...";
    this.modelsReady = false;
    try {
      const [yolo, embed] = await Promise.all([
        ModelLoader.loadModel(YOLO_CONFIG.modelUrl),
        ModelLoader.loadModel(EMBEDDING_CONFIG.modelUrl),
      ]);
      this.yoloModel = yolo;
      this.embedModel = embed;
      this.yoloProcessor = new YoloProcessor(this.yoloModel, YOLO_CONFIG);
      this.embeddingProcessor = new EmbeddingProcessor(
        this.embedModel,
        EMBEDDING_CONFIG,
      );
      this.modelsReady = true;
      this.inferenceStatus = "Models loaded.";
      this.checkAndStartLoops();
    } catch (error: any) {
      this.handleFatalError("Model load failed", error);
      this.modelsReady = false;
    }
  }
  private async startVideoStream() {
    /* ... (No changes needed from V3) ... */
    if (!this.videoElement) return;
    this.inferenceStatus = "Requesting camera...";
    this.streamReady = false;
    try {
      await this.videoSourceManager.startStream(this.videoElement);
      this.streamReady = true;
      this.inferenceStatus = "Video stream started.";
      this.canvasRenderer?.updateDimensions(this.videoElement);
      this.checkAndStartLoops();
    } catch (error: any) {
      this.inferenceStatus =
        this.videoSourceManager.getError() || "Camera failed.";
      this.streamReady = false;
    }
  }
  private handleVideoMetadataLoaded = () => {
    /* ... (No changes needed from V3) ... */
    if (this.canvasRenderer) {
      const vid = this.videoSourceManager?.getVideoElement();
      if (vid) this.canvasRenderer.updateDimensions(vid);
    }
    this.checkAndStartLoops();
  };
  private checkAndStartLoops() {
    /* ... (No changes needed from V3) ... */
    if (
      this.modelsReady &&
      this.streamReady &&
      this.videoSourceManager.isReady()
    ) {
      /* console.log("Starting loops."); */ this.inferenceStatus =
        "Starting...";
      this.startInferenceLoop();
      this.startEmbeddingLoop();
    } else {
      if (!this.modelsReady) this.inferenceStatus = "Waiting models...";
      else if (!this.streamReady) this.inferenceStatus = "Waiting camera...";
      else if (!this.videoSourceManager.isReady())
        this.inferenceStatus = "Waiting video...";
    }
  }
  private startInferenceLoop() {
    /* ... (No changes needed from V3) ... */ if (
      this.inferenceLoopId === null &&
      this.modelsReady &&
      this.streamReady
    ) {
      /* console.log("Start inference loop"); */ this.isInferencing = false;
      this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
    }
  }
  private startEmbeddingLoop() {
    /* ... (No changes needed from V3) ... */ if (
      this.embeddingLoopTimeoutId === null &&
      this.modelsReady &&
      this.streamReady
    ) {
      /* console.log("Start embed loop"); */ this.isEmbedding = false;
      this.embeddingLoopTimeoutId = window.setTimeout(
        () => this.runEmbeddingLoop(),
        EMBEDDING_LOOP_INTERVAL_MS,
      );
    }
  }
  private stopLoops() {
    /* ... (No changes needed from V3) ... */ if (
      this.inferenceLoopId !== null
    ) {
      cancelAnimationFrame(this.inferenceLoopId);
      this.inferenceLoopId = null;
    }
    if (this.embeddingLoopTimeoutId !== null) {
      clearTimeout(this.embeddingLoopTimeoutId);
      this.embeddingLoopTimeoutId = null;
    }
    this.isInferencing = false;
    this.isEmbedding = false;
  }

  // --- Main Inference Loop ---
  private async runInference() {
    if (this.isInferencing) {
      return;
    }
    if (!this._checkInferencePrerequisites()) {
      // Use check function
      this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
      return;
    }

    this.isInferencing = true;
    const frameStart = performance.now();
    const videoElement = this.videoSourceManager.getVideoElement()!;
    const videoDims = this.videoSourceManager.getVideoDimensions()!;
    this.canvasRenderer.updateDimensions(videoElement);

    let yoloOutput: {
      detections: RawDetection[];
      protoTensor: tf.Tensor | null;
    } | null = null;
    let generatedMasks: Map<number, tf.Tensor2D> | null = null;

    try {
      // Execute pipeline steps using helpers
      yoloOutput = await this._getFrameAndYoloOutput(videoElement);
      if (!yoloOutput || !yoloOutput.protoTensor)
        throw new Error("YOLO processing failed.");

      const trackingResults = this._updateTrackerAndGetResults(yoloOutput);

      const maskGenConfig = {
        protoSize: YOLO_CONFIG.protoMaskSize,
        coeffCount: YOLO_CONFIG.maskCoeffCount,
        paddingFactor: DRAWING_CONFIG.maskCropPaddingFactor,
        threshold: DRAWING_CONFIG.maskThreshold,
      };
      generatedMasks = await this._generateMasksForFrame(
        yoloOutput,
        trackingResults,
        videoDims,
        maskGenConfig,
      );

      this._updateTrackedObjectBBoxes(yoloOutput, trackingResults); // Update bboxes in state manager
      this.trackedObjectState.updateTrackedObjects(
        this.objectTracker.getTrackedObjects(),
        frameStart,
      ); // Update state manager's view of objects

      const objectsToDraw = this._prepareDrawData(
        yoloOutput,
        trackingResults,
        generatedMasks,
      );

      await this.canvasRenderer.drawFrame(objectsToDraw);

      // Update status
      const fps = 1000 / (performance.now() - frameStart);
      const trackedCount = this.objectTracker
        .getTrackedObjects()
        .filter((o) => o.id !== -1).length;
      this.inferenceStatus = `Tracking ${trackedCount} | FPS: ${fps.toFixed(1)}`;
    } catch (error) {
      console.error("Error in inference loop:", error);
      this.inferenceStatus = `Error: ${error instanceof Error ? error.message : error}`;
      this.canvasRenderer?.clearAll();
    } finally {
      this._cleanupInferenceResources(yoloOutput, generatedMasks); // Cleanup inter-stage tensors
      this.isInferencing = false;
      if (this.inferenceLoopId !== null) {
        this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
      }
    }
  }

  // --- runInference Helper Methods ---

  private _checkInferencePrerequisites(): boolean {
    /* ... (No changes needed from V3) ... */ return (
      this.modelsReady &&
      this.streamReady &&
      this.videoSourceManager.isReady() &&
      this.yoloProcessor &&
      this.canvasRenderer &&
      this.objectTracker &&
      this.trackedObjectState
    );
  }

  private async _getFrameAndYoloOutput(
    videoElement: HTMLVideoElement,
  ): Promise<{ detections: RawDetection[]; protoTensor: tf.Tensor | null }> {
    /* ... (No changes needed from V3) ... */
    let frameTensor: tf.Tensor | null = null;
    try {
      frameTensor = tf.browser.fromPixels(videoElement);
      const output = await this.yoloProcessor.processFrame(frameTensor);
      return output;
    } finally {
      frameTensor?.dispose();
    }
  }

  private _updateTrackerAndGetResults(yoloOutput: {
    detections: RawDetection[];
  }): number[] {
    /* ... (No changes needed from V3) ... */
    const trackerInputs = yoloOutput.detections.map((det) => ({
      point: [
        (det.bboxModel[0] + det.bboxModel[2]) / 2,
        (det.bboxModel[1] + det.bboxModel[3]) / 2,
      ] as Point,
    }));
    return this.objectTracker.update(trackerInputs);
  }

  // ** RESTORED/CORRECTED HELPER **
  // Scales MODEL coordinates (0-INPUT_W/H) to VIDEO coordinates (0-videoW/H)
  private scaleModelBboxToVideo(bboxModel: number[]): BBox | null {
    const dims = this.videoSourceManager.getVideoDimensions();
    if (!dims || bboxModel.length < 4) return null;
    const [x1_m, y1_m, x2_m, y2_m] = bboxModel;
    // Ensure config values are accessible or pass them
    const inputWidth = YOLO_CONFIG.inputWidth;
    const inputHeight = YOLO_CONFIG.inputHeight;
    return {
      x1: (x1_m / inputWidth) * dims.w,
      y1: (y1_m / inputHeight) * dims.h,
      x2: (x2_m / inputWidth) * dims.w,
      y2: (y2_m / inputHeight) * dims.h,
    };
  }

  // Uses the state manager to update bboxes
  private _updateTrackedObjectBBoxes(
    yoloOutput: { detections: RawDetection[] },
    trackingResults: number[],
  ): void {
    yoloOutput.detections.forEach((rawDet, index) => {
      const trackId = trackingResults[index];
      if (trackId !== -1) {
        const videoBbox = this.scaleModelBboxToVideo(rawDet.bboxModel); // USE THE RESTORED FUNCTION
        this.trackedObjectState.updateLastKnownBbox(trackId, videoBbox);
      }
    });
  }

  private async _generateMasksForFrame(
    yoloOutput: { detections: RawDetection[]; protoTensor: tf.Tensor | null },
    trackingResults: number[],
    videoDims: VideoDims,
    maskGenConfig: any,
  ): Promise<Map<number, tf.Tensor2D>> {
    /* ... (No changes needed from V3, uses functional generateMask) ... */
    const generatedMasks = new Map<number, tf.Tensor2D>();
    const maskGenPromises: Promise<void>[] = [];
    if (!yoloOutput.protoTensor) return generatedMasks;
    yoloOutput.detections.forEach((rawDet, index) => {
      const videoBbox = this.scaleModelBboxToVideo(rawDet.bboxModel);
      if (videoBbox) {
        const p = generateMask(
          yoloOutput.protoTensor!,
          rawDet.maskCoeffs,
          videoBbox,
          videoDims,
          maskGenConfig,
        ).then((mT) => {
          if (mT) {
            generatedMasks.set(index, mT);
          }
        });
        maskGenPromises.push(p);
      }
    });
    await Promise.all(maskGenPromises);
    return generatedMasks;
  }

  private _prepareDrawData(
    yoloOutput: { detections: RawDetection[] },
    trackingResults: number[],
    generatedMasks: Map<number, tf.Tensor2D> | null,
  ): DrawInfo[] {
    /* ... (No changes needed from V3, uses restored scaleModelBboxToVideo indirectly) ... */
    const objectsToDraw: DrawInfo[] = [];
    yoloOutput.detections.forEach((rawDet, index) => {
      const trackId = trackingResults[index];
      const videoBbox = this.scaleModelBboxToVideo(rawDet.bboxModel);
      if (videoBbox) {
        const canvasBbox =
          this.canvasRenderer.convertVideoBoxToCanvas(videoBbox);
        const label = `${trackId === -1 ? "Init" : `ID: ${trackId}`} (${rawDet.confidence.toFixed(2)})`;
        const color =
          DRAWING_CONFIG.colors[
            trackId >= 0
              ? trackId % DRAWING_CONFIG.colors.length
              : DRAWING_CONFIG.colors.length - 1
          ];
        const maskTensor = generatedMasks?.get(index);
        objectsToDraw.push({
          id: trackId,
          color: color,
          bboxCanvas: canvasBbox,
          label: label,
          mask: maskTensor
            ? { tensor: maskTensor, videoBox: videoBbox }
            : undefined,
        });
      }
    });
    return objectsToDraw;
  }

  private _cleanupInferenceResources(
    yoloOutput: {
      detections: RawDetection[];
      protoTensor: tf.Tensor | null;
    } | null,
    generatedMasks: Map<number, tf.Tensor2D> | null,
  ): void {
    /* ... (No changes needed from V3) ... */
    yoloOutput?.detections.forEach((det) => det.maskCoeffs.dispose());
    yoloOutput?.protoTensor?.dispose();
    generatedMasks?.forEach((mask) => mask.dispose());
  }

  // --- Embedding Loop (Uses TrackedObjectState) ---
  private async runEmbeddingLoop() {
    /* ... (No changes needed from V3, uses state manager correctly) ... */
    if (this.isEmbedding) {
      return;
    }
    if (
      !this.modelsReady ||
      !this.streamReady ||
      !this.videoSourceManager.isReady() ||
      !this.embeddingProcessor ||
      !this.trackedObjectState
    ) {
      this.embeddingLoopTimeoutId = window.setTimeout(
        () => this.runEmbeddingLoop(),
        EMBEDDING_LOOP_INTERVAL_MS,
      );
      return;
    }
    this.isEmbedding = true;
    let embeddingCandidate: { id: number; bboxVideo: BBox } | null = null;
    try {
      embeddingCandidate = this.trackedObjectState.selectObjectForEmbedding();
      if (embeddingCandidate) {
        const { id, bboxVideo } = embeddingCandidate;
        const videoElement = this.videoSourceManager.getVideoElement()!;
        const newEmbedding = await this.embeddingProcessor.createEmbedding(
          videoElement,
          bboxVideo,
        );
        if (newEmbedding) {
          this.trackedObjectState.updateEmbedding(id, newEmbedding);
          // TODO: need some way to know if the embedding is new
          this.fetchDataFromVectorDB(newEmbedding, id).then((uid) => {
            if (uid) {
              this.trackedObjectState.updateMatchingId(uid);
              console.log(`Embedding for ID ${id} stored with UID: ${uid}`);
            } else {
              console.warn(`No UID found for ID ${id}`);
            }
          });
        }
      }
    } catch (error) {
      console.error(`Error embedding (ID: ${embeddingCandidate?.id}):`, error);
    } finally {
      this.isEmbedding = false;
      if (this.embeddingLoopTimeoutId !== null) {
        this.embeddingLoopTimeoutId = window.setTimeout(
          () => this.runEmbeddingLoop(),
          EMBEDDING_LOOP_INTERVAL_MS,
        );
      }
    }
  }

  // --- Other Helpers ---
  private async fetchDataFromVectorDB(embedding: tf.Tensor, objectId: number) {
    if (!indexProcessVec || !index) {
      return;
    }
    const embArray = Array.from(await embedding.data());
    const compressedEmbedding = indexProcessVec(embArray);
    const results = index.searchKNN(compressedEmbedding, 1);
    const result = results[0];
    return result.id;
  }
  private handleFatalError(message: string, error?: any) {
    /* ... (No changes needed from V3) ... */ console.error(
      "FATAL ERROR:",
      message,
      error,
    );
    this.inferenceStatus = `FATAL: ${message}`;
    this.stopLoops();
    this.videoSourceManager?.stopStream();
    this.modelsReady = false;
    this.streamReady = false;
    this.canvasRenderer?.clearAll();
  }

  // --- Render ---
  render() {
    /* ... (No changes needed from V3) ... */ return html`<video
        id="video"
        muted
        playsinline
      ></video
      ><canvas id="overlay-canvas"></canvas>
      <div class="status">${this.inferenceStatus}</div>`;
  }
} // End VideoContainer class

// ===========================================================
// Entry Point / Main Function (Minimal Change)
// ===========================================================

export function main() {
  /* ... (No changes needed from V3) ... */
  if (!customElements.get("video-container")) {
    customElements.define("video-container", VideoContainer);
  }
  document.querySelector("video-container")?.remove();
  const videoContainer = document.createElement("video-container");
  document.body.appendChild(videoContainer);

  // load the index, call async loadIndex
  loadIndex()
    .then((index) => {
      console.log("Index loaded:", index);
      indexLoadState = "Success";
    })
    .catch((error) => {
      console.error("Error loading index:", error);
      indexLoadState = "Error";
    });
}
