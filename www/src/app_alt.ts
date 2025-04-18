import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, state, query, customElement } from 'lit/decorators.js';
import * as tf from '@tensorflow/tfjs';

// Import Norfair types and classes
import {Tracker, TrackedObject, TrackerOptions, Point, Detection} from "./norfair";

// Type for storing embedding status and result for each tracked object
type ObjectEmbeddingInfo = {
    id: number;
    hasBeenEmbedded: boolean;
    lastEmbeddingTime: number | null;
    creationTime: number;
    embedding: tf.Tensor | null; // Store the embedding tensor
    // Store the last known bounding box in *video coordinates* for cropping
    lastKnownBboxVideo: { x1: number, y1: number, x2: number, y2: number } | null;
};

// ===========================================================
// App Container Component
// ===========================================================

export function main() {
  const appContainer = document.createElement('app-container');
  document.body.appendChild(appContainer);
}


@customElement('app-container')
class AppContainer extends LitElement {
  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: sans-serif;
    }
    video-container {
      flex: 1;
      min-height: 0;
    }
  `;

  render() {
    return html`<video-container></video-container>`;
  }
}

interface DetectionResult {
  bboxModel: number[]; // Full detection data
  point: Point;
  maskTensor?: tf.Tensor2D; // Optional mask tensor
}

// ===========================================================
// Video Container Component
// ===========================================================

// --- YOLO Configuration ---
const MODEL_URL = '/assets/models/yolov11s_seg__dk964hap__web_model/model.json';
const MODEL_INPUT_WIDTH = 640;
const MODEL_INPUT_HEIGHT = 640;
const CONFIDENCE_THRESHOLD = 0.5;
const MASK_THRESHOLD = 0.5; // Threshold for converting sigmoid output to binary mask
const PROTO_MASK_SIZE = 160; // Assumed dimension (e.g., 160x160) of prototype masks - VERIFY THIS
const MASK_COEFF_COUNT = 32; // Assumed number of mask coefficients - VERIFY THIS
// --- EMBED Configuration ---
const EMBED_INPUT_WIDTH = 128;
const EMBED_INPUT_HEIGHT = 192;
const EMBED_URL = '/assets/models/convnextv2_convlinear__aivb8jvk-47500__encoder__web_model/model.json';
const EMBEDDING_LOOP_INTERVAL_MS = 250; // How often to check for embedding tasks
const EMBEDDING_CROP_PADDING_FACTOR = 0.1; // Add 10% padding around bbox for crop
// --- Color Palette ---
const COLORS = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    [255, 0, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
    [128, 0, 128], [0, 128, 128], [255, 165, 0], [255, 192, 203], [75, 0, 130]
];
// ---------------------

@customElement('video-container')
class VideoContainer extends LitElement {
  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      position: relative;
      width: 100%;
      height: 100%;
      background-color: #222;
      overflow: hidden;
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
      padding: 5px 10px;
      background-color: rgba(0, 0, 0, 0.7);
      color: white;
      font-size: 0.9em;
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      text-align: center;
      z-index: 10;
    }
  `;

  // --- Element References ---
  private ctx!: CanvasRenderingContext2D;
  @query('#video')
  private videoElement!: HTMLVideoElement;
  @query('#overlay-canvas')
  private canvasElement!: HTMLCanvasElement;

  // --- TFJS State ---
  @state() private tfjsModel: tf.GraphModel | null = null; // YOLO model
  @state() private isModelLoading: boolean = true;
  @state() private modelError: string | null = null;
  @state() private tfjsEmbedModel: tf.GraphModel | null = null; // Embedding model
  @state() private isEmbedModelLoading: boolean = true;
  @state() private embedModelError: string | null = null;
  @state() private inferenceStatus: string = 'Initializing...';

  // --- Video Stream State ---
  private currentStream: MediaStream | null = null;
  private inferenceLoopId: number | null = null; // For YOLO + Tracking
  private videoDisplayWidth: number = 0;
  private videoDisplayHeight: number = 0;
  private resizeObserver!: ResizeObserver;
  private isInferencing: boolean = false; // For YOLO + Tracking loop
  private scale: number = 1;
  private offsetX: number = 0;
  private offsetY: number = 0;

  // --- Tracking state ---
  private tracker: Tracker | null = null;
  // --- Embedding State ---
  private objectEmbeddingStatus = new Map<number, ObjectEmbeddingInfo>(); // Map<trackId, info>
  private embeddingLoopTimeoutId: number | null = null; // For Embedding loop
  private isEmbedding: boolean = false; // For Embedding loop

  // --- Canvas for mask rendering ---
  private maskCanvas!: HTMLCanvasElement;
  private maskCtx!: CanvasRenderingContext2D;

  // --- Lit Lifecycle ---

  connectedCallback() {
    super.connectedCallback();
    // Create offscreen canvas for mask rendering
    this.maskCanvas = document.createElement('canvas');
    const context = this.maskCanvas.getContext('2d', { willReadFrequently: true }); // willReadFrequently might be needed for frequent toPixels/putImageData
     if (!context) {
        console.error("Failed to get 2D context for mask canvas");
        // Handle error appropriately, maybe disable mask rendering
    } else {
         this.maskCtx = context;
    }

    tf.ready().then(async () => {
      this.loadTfjsModels();
      this.startVideoStream();
      // Initialize the Norfair tracker
      this.tracker = new Tracker({
          distanceThreshold: 100,
          hitInertiaMin: 0,
          hitInertiaMax: 15,
          initDelay: 2,
      });
      console.log("Norfair tracker initialized.");
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.stopVideoStream(); // Stops inference loop too
    this.cancelEmbeddingLoop(); // Stop embedding loop
    this.tfjsModel?.dispose();
    this.tfjsEmbedModel?.dispose();
    this.tfjsModel = null;
    this.tfjsEmbedModel = null;
    this.resizeObserver?.disconnect();
    this.tracker = null;
    // Dispose of any remaining stored embeddings
    this.objectEmbeddingStatus.forEach(info => info.embedding?.dispose());
    this.objectEmbeddingStatus.clear();
  }

  protected firstUpdated(_changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>): void {
    if (!this.videoElement || !this.canvasElement) {
      this.handleFatalError("Initialization Error: Video or Canvas element missing.");
      return;
    }
    const context = this.canvasElement.getContext('2d');
    if (!context) {
      this.handleFatalError("Canvas context error.");
      return;
    }
    this.ctx = context;

    this.videoElement.addEventListener('loadedmetadata', this.handleVideoMetadataLoaded);
    this.videoElement.addEventListener('error', (e) => this.handleFatalError('Video element error.', e));

    this.resizeObserver = new ResizeObserver(() => this.updateCanvasSize());
    this.resizeObserver.observe(this.videoElement);
    this.updateCanvasSize();
  }

  // --- Event Handlers ---
  private handleVideoMetadataLoaded = () => {
    this.videoElement.play().catch(e => this.handleFatalError('Video play failed:', e));
    this.updateCanvasSize();
    // Start loops only when models are loaded AND video metadata is ready
    this.startInferenceLoop();
    this.startEmbeddingLoop();
  }

  private handleFatalError(message: string, error?: any) {
      console.error(message, error);
      this.inferenceStatus = message;
      this.stopVideoStream();
      this.cancelEmbeddingLoop();
      this.requestUpdate();
  }

  // --- Size & Scaling ---
  private updateCanvasSize() {
    if (!this.videoElement || !this.canvasElement || !this.videoElement.videoWidth) {
      return;
    }
    const displayWidth = this.videoElement.clientWidth;
    const displayHeight = this.videoElement.clientHeight;

    if (this.canvasElement.width !== displayWidth || this.canvasElement.height !== displayHeight) {
        this.canvasElement.width = displayWidth;
        this.canvasElement.height = displayHeight;
        // Resize mask canvas too
        if (this.maskCanvas) {
            this.maskCanvas.width = displayWidth;
            this.maskCanvas.height = displayHeight;
        }
        this.videoDisplayWidth = displayWidth;
        this.videoDisplayHeight = displayHeight;
    }

    // ... (calculate scale and offset - no changes) ...
    const videoWidth = this.videoElement.videoWidth;
    const videoHeight = this.videoElement.videoHeight;
    const scaleX = this.videoDisplayWidth / videoWidth;
    const scaleY = this.videoDisplayHeight / videoHeight;
    this.scale = Math.min(scaleX, scaleY);
    this.offsetX = (this.videoDisplayWidth - videoWidth * this.scale) / 2;
    this.offsetY = (this.videoDisplayHeight - videoHeight * this.scale) / 2;
  }


  // --- TFJS Methods ---
  private async loadTfjsModels() {
    this.isModelLoading = true; this.isEmbedModelLoading = true;
    this.modelError = null; this.embedModelError = null;
    this.inferenceStatus = 'Loading models...'; this.requestUpdate();

    try {
      // Load models in parallel
      const [yoloModel, embedModel] = await Promise.all([
        tf.loadGraphModel(MODEL_URL),
        tf.loadGraphModel(EMBED_URL)
      ]);

      this.tfjsModel = yoloModel;
      this.isModelLoading = false;
      console.log('YOLO model loaded successfully.');

      this.tfjsEmbedModel = embedModel;
      this.isEmbedModelLoading = false;
      console.log('Embedding model loaded successfully.');

      this.inferenceStatus = 'Models loaded. Waiting for video...';

    } catch (error: any) {
        console.error('TFJS model load failed:', error);
        const errorMessage = `Model load failed: ${error.message || error}`;
        if (!this.tfjsModel) {
            this.modelError = errorMessage;
            this.isModelLoading = false;
        }
        if (!this.tfjsEmbedModel) {
            this.embedModelError = errorMessage;
            this.isEmbedModelLoading = false;
        }
        this.inferenceStatus = errorMessage;
    } finally {
        this.requestUpdate();
        if(this.videoElement && this.videoElement.readyState >= this.videoElement.HAVE_METADATA) {
            this.startInferenceLoop();
            this.startEmbeddingLoop();
        }
    }
  }

  // --- Video Stream Methods ---
  private async startVideoStream() {
    this.inferenceStatus = 'Requesting camera...'; this.requestUpdate();
    if (this.currentStream) {
      this.stopVideoStream();
    }
    try {
      const constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 } } };
      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      if (this.videoElement) {
        this.videoElement.srcObject = this.currentStream;
        const loadingStatus = (this.isModelLoading || this.isEmbedModelLoading) ? 'Models loading...' : 'Models loaded.';
        this.inferenceStatus = `Video stream started. ${loadingStatus}`;
      } else {
        this.inferenceStatus = 'Video element not ready.';
      }
    } catch (error: any) {
      this.handleFatalError(`Camera failed: ${error.message}`, error);
    } finally {
      this.requestUpdate();
    }
  }

  private stopVideoStream() {
    this.cancelInferenceLoop();
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop());
    }
    this.currentStream = null;
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.srcObject = null;
    }
    this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
    if (!this.modelError && !this.embedModelError) {
        this.inferenceStatus = "Video stream stopped.";
    }
    this.requestUpdate();
  }

  // --- Inference Loop (YOLO + Tracking) ---
  private startInferenceLoop() {
    if (this.inferenceLoopId !== null || this.isModelLoading || !this.currentStream || !this.tfjsModel || !this.videoElement || this.videoElement.readyState < this.videoElement.HAVE_METADATA) {
      return;
    }
    console.log("Starting YOLO/Tracking inference loop...");
    this.isInferencing = false;
    this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
  }

  private cancelInferenceLoop() {
    if (this.inferenceLoopId !== null) {
      cancelAnimationFrame(this.inferenceLoopId);
      this.inferenceLoopId = null;
      this.isInferencing = false;
      if (!this.modelError && !this.embedModelError) {
        this.inferenceStatus = "Inference stopped.";
        this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
        this.requestUpdate();
      }
      console.log("YOLO/Tracking inference loop cancelled.");
    }
  }

  private async runInference() {
    if (this.isInferencing) { /* ... skip ... */ return; }
    if (!this.tfjsModel || !this.currentStream || !this.videoElement || !this.ctx || this.videoElement.paused || this.videoElement.ended || this.videoElement.readyState < this.videoElement.HAVE_CURRENT_DATA || !this.tracker) { /* ... wait ... */ return; }

    this.isInferencing = true;
    this.updateCanvasSize();

    const tensorsToDispose: tf.Tensor[] = [];
    let output: tf.Tensor[] | null = null;
    const frameTime = Date.now();

    try {
        const frameTensor = tf.browser.fromPixels(this.videoElement);
        tensorsToDispose.push(frameTensor);

        const inputTensor = tf.tidy(() => { /* ... preprocess frame ... */
            const resized = tf.image.resizeBilinear(frameTensor, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]);
            const normalized = resized.div(255.0);
            const batched = normalized.expandDims(0);
            return batched.cast('float32');
        });

        // CORRECT Output Shapes:
        // output[0].shape == [1, 300, 38]   (batch, num_dets, bbox[4]+conf[1]+class[1]+coeffs[32])
        // output[1].shape == [1500]        (Not used in this logic)
        // output[2].shape == [1, 160, 160, 32] (batch, proto_height, proto_width, num_coeffs)
        output = await this.tfjsModel!.executeAsync(inputTensor) as tf.Tensor[];

        if (output) {
          tensorsToDispose.push(...output);
        }

        // --- Process Detections and Masks ---
        // *** Check output indices and shapes based on your specific model ***
        if (output && output.length >= 3) { // Need detections (output[0]) and protos (output[2])
            const detectionsTensor = output[0]; // Shape [1, num_dets, 38]
            const protoTensor = output[2];      // Shape [1, 160, 160, 32] <-- CORRECTED SHAPE

            // --- CORRECTED Shape Validation ---
            // Validate the dimensions of the prototype mask tensor
             if (protoTensor.shape.length !== 4 ||
                 protoTensor.shape[0] !== 1 ||        // Batch size
                 protoTensor.shape[1] !== PROTO_MASK_SIZE || // Height
                 protoTensor.shape[2] !== PROTO_MASK_SIZE || // Width
                 protoTensor.shape[3] !== MASK_COEFF_COUNT) { // Number of coefficients
                  console.error("Unexpected prototype mask tensor shape:", protoTensor.shape, "Expected: [1,", PROTO_MASK_SIZE, ",", PROTO_MASK_SIZE, ",", MASK_COEFF_COUNT, "]");
                  throw new Error("Incorrect prototype mask shape.");
             }
             // --- END CORRECTED Shape Validation ---


            const detectionsBatch = (await detectionsTensor.data()) as Float32Array; // Use .data() for direct access
            const numDets = detectionsTensor.shape[1];
            const detDataLength = detectionsTensor.shape[2]; // Length of one detection vector (38)

            const norfairDetections: Point[] = [];
            const currentFrameDetectionResults = new Map<number, DetectionResult>(); // Map norfair index -> result

            for (let i = 0; i < numDets; i++) {
                const offset = i * detDataLength;
                const confidence = detectionsBatch[offset + 4];

                if (confidence >= CONFIDENCE_THRESHOLD) {
                    const x1 = detectionsBatch[offset + 0];
                    const y1 = detectionsBatch[offset + 1];
                    const x2 = detectionsBatch[offset + 2];
                    const y2 = detectionsBatch[offset + 3];
                    const classId = Math.round(detectionsBatch[offset + 5]);
                    // Extract mask coefficients (last MASK_COEFF_COUNT elements)
                    // Slice from index 6 to get the 32 coefficients
                    const maskCoeffs = tf.slice(detectionsTensor, [0, i, 6], [1, 1, MASK_COEFF_COUNT]); // Shape [1, 1, 32]

                    const centerX = (x1 + x2) / 2;
                    const centerY = (y1 + y2) / 2;
                    const point: Point = [centerX, centerY];
                    const norfairIndex = norfairDetections.length;
                    norfairDetections.push(point);

                    // --- Calculate Mask ---
                    const binaryMask = tf.tidy(() => {
                        // --- CORRECTED Mask Calculation Logic ---
                        // protoTensor shape: [1, H, W, C] = [1, 160, 160, 32]
                        // maskCoeffs shape: [1, 1, 32]

                        // 1. Reshape protos: [1, H, W, C] -> [H*W, C] = [25600, 32]
                        const protosReshaped = protoTensor.squeeze(0).reshape([PROTO_MASK_SIZE * PROTO_MASK_SIZE, MASK_COEFF_COUNT]);

                        // 2. Reshape coefficients: [1, 1, 32] -> [32, 1]
                        const coeffsReshaped = maskCoeffs.reshape([MASK_COEFF_COUNT, 1]);

                        // 3. Matrix Multiply: [H*W, C] @ [C, 1] -> [H*W, 1] = [25600, 1]
                        // This performs the weighted sum of prototypes for each pixel location.
                        const maskProto = tf.matMul(protosReshaped, coeffsReshaped);

                        // 4. Reshape back to spatial: [H*W, 1] -> [H, W] = [160, 160]
                        const maskReshaped = maskProto.reshape([PROTO_MASK_SIZE, PROTO_MASK_SIZE]);
                        // --- END CORRECTED Mask Calculation Logic ---

                        // Activation (Sigmoid)
                        const maskActivated = tf.sigmoid(maskReshaped);

                        // --- Upscale and Crop Mask (No change needed in this part) ---
                        const videoBbox = this.scaleModelBboxToVideo(detectionsBatch.slice(offset, offset + 4));
                        if (!videoBbox) return null;

                        const videoWidth = this.videoElement.videoWidth;
                        const videoHeight = this.videoElement.videoHeight;

                        const boxX1 = Math.max(0, Math.floor(videoBbox.x1));
                        const boxY1 = Math.max(0, Math.floor(videoBbox.y1));
                        const boxX2 = Math.min(videoWidth, Math.ceil(videoBbox.x2));
                        const boxY2 = Math.min(videoHeight, Math.ceil(videoBbox.y2));
                        const boxW = boxX2 - boxX1;
                        const boxH = boxY2 - boxY1;

                        if (boxW <= 0 || boxH <= 0) return null;

                        const normalizedBbox = [[
                            boxY1 / videoHeight, boxX1 / videoWidth,
                            boxY2 / videoHeight, boxX2 / videoWidth
                        ]];

                        const maskExpanded = maskActivated.expandDims(0).expandDims(-1); // [1, H, W, 1]

                        const maskCroppedResized = tf.image.cropAndResize(
                            maskExpanded, normalizedBbox, [0], [boxH, boxW], 'bilinear'
                        );

                        const finalMask = maskCroppedResized.squeeze([0, 3]).greater(MASK_THRESHOLD).cast('float32'); // Binary mask [boxH, boxW]

                        return finalMask as tf.Tensor2D;
                    }); // End tf.tidy for mask calculation

                    // Store result
                    currentFrameDetectionResults.set(norfairIndex, {
                        bboxModel: Array.from(detectionsBatch.slice(offset, offset + detDataLength)),
                        point: point,
                        maskTensor: binaryMask || undefined
                    });

                     // Dispose maskCoeffs immediately as it's sliced from a larger tensor
                     maskCoeffs.dispose();

                } // End confidence check
            } // End detection loop

            // --- Update Tracker ---
            const trackingResults = this.tracker.update(norfairDetections);
            this.updateObjectStatus(this.tracker.trackedObjects, frameTime);

            // --- Drawing Loop (No changes needed here, relies on correct binaryMask) ---
            this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
            let trackedDetectionCount = 0;
            this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
            const maskPromises: Promise<void>[] = [];

            for (let i = 0; i < norfairDetections.length; i++) {
                const assignedId = trackingResults[i];
                const detResult = currentFrameDetectionResults.get(i);

                if (detResult) {
                    trackedDetectionCount++;
                    const colorIndex = assignedId >= 0 ? assignedId % COLORS.length : COLORS.length - 1;
                    const color = COLORS[colorIndex];

                    this.drawBoundingBoxWithId(detResult.bboxModel, color, assignedId);

                    if (detResult.maskTensor) {
                         maskPromises.push(this.drawSegmentationMask(detResult.maskTensor, color, detResult.bboxModel));
                    }

                    if (assignedId !== -1 && this.objectEmbeddingStatus.has(assignedId)) {
                        const videoBbox = this.scaleModelBboxToVideo(detResult.bboxModel);
                        if (videoBbox) {
                             this.objectEmbeddingStatus.get(assignedId)!.lastKnownBboxVideo = videoBbox;
                        }
                    }
                } else {
                     currentFrameDetectionResults.get(i)?.maskTensor?.dispose();
                }
            }
            await Promise.all(maskPromises);
            this.ctx.save();
            this.ctx.globalAlpha = 0.5;
            this.ctx.drawImage(this.maskCanvas, 0, 0);
            this.ctx.restore();

            this.inferenceStatus = `Tracked Detections: ${trackedDetectionCount}`;

        } else { // Handle case where output doesn't have enough elements
             console.warn("YOLO output missing expected tensors:", output?.length);
             this.inferenceStatus = "Model output missing tensors.";
             this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
             this.tracker.update([]);
             this.updateObjectStatus(this.tracker.trackedObjects, frameTime);
        }

    } catch (error: any) {
        console.error("Error during YOLO/Tracking/Masking inference:", error);
        this.inferenceStatus = `Inference Error: ${error.message || error}`;
        this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
        if (this.tracker) { /* ... update tracker on error ... */ }
    } finally {
        tf.dispose(tensorsToDispose);
        this.isInferencing = false;
        this.requestUpdate();
        if (this.inferenceLoopId !== null) {
             this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
        }
    }
  }


   // --- Embedding Loop ---
   private startEmbeddingLoop() {
    if (this.embeddingLoopTimeoutId !== null || this.isEmbedModelLoading || !this.currentStream || !this.tfjsEmbedModel || !this.videoElement || this.videoElement.readyState < this.videoElement.HAVE_METADATA) {
        return;
    }
    console.log("Starting Embedding loop...");
    this.isEmbedding = false;
    this.embeddingLoopTimeoutId = window.setTimeout(() => this.runEmbeddingLoop(), EMBEDDING_LOOP_INTERVAL_MS);
   }

   private cancelEmbeddingLoop() {
    if (this.embeddingLoopTimeoutId !== null) {
        clearTimeout(this.embeddingLoopTimeoutId);
        this.embeddingLoopTimeoutId = null;
        this.isEmbedding = false;
        console.log("Embedding loop cancelled.");
    }
   }

    private async runEmbeddingLoop() {
        if (this.isEmbedding) {
            console.log("Already embedding, skipping cycle.");
            this.embeddingLoopTimeoutId = window.setTimeout(() => this.runEmbeddingLoop(), EMBEDDING_LOOP_INTERVAL_MS);
            return;
        }
        if (!this.tfjsEmbedModel || !this.currentStream || !this.videoElement || this.videoElement.paused || this.videoElement.ended || this.videoElement.readyState < this.videoElement.HAVE_CURRENT_DATA || !this.tracker) {
             this.embeddingLoopTimeoutId = window.setTimeout(() => this.runEmbeddingLoop(), EMBEDDING_LOOP_INTERVAL_MS);
             return;
        }

        this.isEmbedding = true;

        let finalEmbeddingTensor: tf.Tensor | null = null; // To hold the final result if successful
        let objectIdForEmbedding: number = -1;

        try {
            const objectToEmbed = this.selectObjectForEmbedding();

            if (objectToEmbed && this.objectEmbeddingStatus.has(objectToEmbed.id)) {
                 objectIdForEmbedding = objectToEmbed.id; // Store ID for later use
                 const objectInfo = this.objectEmbeddingStatus.get(objectIdForEmbedding)!;
                 const bboxVideo = objectInfo.lastKnownBboxVideo;

                 if (bboxVideo) {
                    console.log(`Embedding object ID: ${objectIdForEmbedding}`);

                    // --- Manual Tensor Management Scope ---
                    let imageTensor: tf.Tensor | null = null;
                    let cropped: tf.Tensor | null = null;
                    let normalized: tf.Tensor | null = null;
                    let embedding: tf.Tensor | null = null;
                    const intermediateTensors: tf.Tensor[] = []; // Track intermediate tensors

                    try {
                        // 1. Crop Image from Video Frame
                        imageTensor = tf.browser.fromPixels(this.videoElement);
                        intermediateTensors.push(imageTensor); // Track

                        const videoWidth = this.videoElement.videoWidth;
                        const videoHeight = this.videoElement.videoHeight;
                        const boxWidth = bboxVideo.x2 - bboxVideo.x1;
                        const boxHeight = bboxVideo.y2 - bboxVideo.y1;
                        const padX = boxWidth * EMBEDDING_CROP_PADDING_FACTOR;
                        const padY = boxHeight * EMBEDDING_CROP_PADDING_FACTOR;
                        const cropX1 = Math.max(0, Math.floor(bboxVideo.x1 - padX));
                        const cropY1 = Math.max(0, Math.floor(bboxVideo.y1 - padY));
                        const cropX2 = Math.min(videoWidth, Math.ceil(bboxVideo.x2 + padX));
                        const cropY2 = Math.min(videoHeight, Math.ceil(bboxVideo.y2 + padY));

                        if (cropX2 <= cropX1 || cropY2 <= cropY1) {
                            throw new Error(`Invalid crop dimensions for object ${objectIdForEmbedding}`);
                        }

                        const boxes = [[ cropY1 / videoHeight, cropX1 / videoWidth, cropY2 / videoHeight, cropX2 / videoWidth ]];
                        const boxIndices = [0];
                        const cropSize: [number, number] = [EMBED_INPUT_HEIGHT, EMBED_INPUT_WIDTH];

                        cropped = tf.image.cropAndResize(
                            imageTensor.expandDims(0).toFloat(),
                            boxes, boxIndices, cropSize, 'bilinear'
                        );
                        intermediateTensors.push(cropped); // Track

                        // 2. Normalize
                        normalized = cropped.div(255.0);
                        intermediateTensors.push(normalized); // Track

                        // 3. Run Embedding Model
                        embedding = await this.tfjsEmbedModel!.executeAsync(normalized) as tf.Tensor;
                        // IMPORTANT: Do NOT track the raw embedding output tensor if we are cloning it.
                        // Let the clone be the final result.

                        if (embedding) {
                            // Clone and detach the result BEFORE disposing intermediates
                            finalEmbeddingTensor = embedding.clone();
                        } else {
                             throw new Error(`Embedding model execution returned null for object ${objectIdForEmbedding}`);
                        }

                    } finally {
                        // Dispose intermediate tensors manually
                        tf.dispose(intermediateTensors);
                        // Dispose the raw embedding output if it exists (it shouldn't be tracked if cloned)
                        if (embedding && !embedding.isDisposed) {
                             embedding.dispose();
                        }
                         // console.log("Intermediate embedding tensors disposed.");
                    }
                    // --- End Manual Tensor Management Scope ---


                    if (finalEmbeddingTensor) {
                         // --- Store Embedding & Update Status ---
                         objectInfo.embedding?.dispose(); // Dispose previous embedding
                         objectInfo.embedding = finalEmbeddingTensor; // Store the new clone
                         objectInfo.hasBeenEmbedded = true;
                         objectInfo.lastEmbeddingTime = Date.now();
                         console.log(`Stored new embedding for object ID: ${objectIdForEmbedding}`);

                         // --- Call Placeholder DB Query (pass the cloned tensor) ---
                         await this.fetchDataFromVectorDB(finalEmbeddingTensor, objectIdForEmbedding);
                    }
                 }
            }

        } catch (error: any) {
            console.error(`Error during embedding loop for object ${objectIdForEmbedding}:`, error);
            // Clean up the final tensor if an error occurred after it was created but before storage/use
            if (finalEmbeddingTensor && !finalEmbeddingTensor.isDisposed){
                 finalEmbeddingTensor.dispose();
            }
        } finally {
            this.isEmbedding = false;
            if (this.embeddingLoopTimeoutId !== null) {
                this.embeddingLoopTimeoutId = window.setTimeout(() => this.runEmbeddingLoop(), EMBEDDING_LOOP_INTERVAL_MS);
            }
        }
    }


    // --- Helper to select the next object for embedding based on priority ---
    private selectObjectForEmbedding(): TrackedObject | null {
        if (!this.tracker) return null;

        const activeObjects = this.tracker.trackedObjects.filter(obj => obj.id !== -1);
        if (activeObjects.length === 0) return null;

        const candidatesInfo = activeObjects
            .map(obj => this.objectEmbeddingStatus.get(obj.id))
            .filter((info): info is ObjectEmbeddingInfo => info !== undefined);

        // Priority 1: Unembedded objects, sorted by creation time (oldest first)
        const unembedded = candidatesInfo
            .filter(info => !info.hasBeenEmbedded)
            .sort((a, b) => a.creationTime - b.creationTime);

        if (unembedded.length > 0) {
            return activeObjects.find(obj => obj.id === unembedded[0].id) || null;
        }

        // Priority 2: Embedded objects, sorted by last embedding time (oldest first)
        const embedded = candidatesInfo
             .filter(info => info.hasBeenEmbedded && info.lastEmbeddingTime !== null)
             .sort((a, b) => (a.lastEmbeddingTime as number) - (b.lastEmbeddingTime as number));

        if (embedded.length > 0) {
             return activeObjects.find(obj => obj.id === embedded[0].id) || null;
        }

        return null;
    }

    // --- Helper to manage object status map ---
    private updateObjectStatus(currentTrackedObjects: TrackedObject[], timestamp: number) {
        if (!this.tracker) return; // Ensure tracker exists
        const currentIds = new Set(currentTrackedObjects.map(obj => obj.id).filter(id => id !== -1));

        // Remove dead objects
        const deadIds: number[] = [];
        for (const id of this.objectEmbeddingStatus.keys()) {
            if (!currentIds.has(id)) {
                deadIds.push(id);
            }
        }
        deadIds.forEach(id => {
            const info = this.objectEmbeddingStatus.get(id);
            console.log(`Removing object ID ${id} from embedding status.`);
            info?.embedding?.dispose(); // Dispose stored embedding
            this.objectEmbeddingStatus.delete(id);
        });

        // Add newly initialized objects
        currentTrackedObjects.forEach(obj => {
            if (obj.id !== -1 && !this.objectEmbeddingStatus.has(obj.id)) {
                console.log(`Adding object ID ${obj.id} to embedding status.`);
                this.objectEmbeddingStatus.set(obj.id, {
                    id: obj.id,
                    hasBeenEmbedded: false,
                    lastEmbeddingTime: null,
                    creationTime: timestamp,
                    embedding: null,
                    lastKnownBboxVideo: null,
                });
            }
        });
    }

    // --- Helper to scale model bbox coordinates to video coordinates ---
    private scaleModelBboxToVideo(bboxModel: number[]): { x1: number, y1: number, x2: number, y2: number } | null {
        if (!this.videoElement || !this.videoElement.videoWidth) return null; // Check videoElement too
        const videoWidth = this.videoElement.videoWidth;
        const videoHeight = this.videoElement.videoHeight;
        const [x1_m, y1_m, x2_m, y2_m] = bboxModel;

        const videoX1 = (x1_m / MODEL_INPUT_WIDTH) * videoWidth;
        const videoY1 = (y1_m / MODEL_INPUT_HEIGHT) * videoHeight;
        const videoX2 = (x2_m / MODEL_INPUT_WIDTH) * videoWidth;
        const videoY2 = (y2_m / MODEL_INPUT_HEIGHT) * videoHeight;

        return { x1: videoX1, y1: videoY1, x2: videoX2, y2: videoY2 };
    }

    // --- Placeholder for Vector DB Query ---
    private async fetchDataFromVectorDB(embedding: tf.Tensor, objectId: number): Promise<void> {
        console.log(`Fetching data for object ID ${objectId} using embedding...`);
        await new Promise(resolve => setTimeout(resolve, 50));

        try {
            const embeddingData = await embedding.data();
            console.log(` -> Simulated DB query complete for ID ${objectId}. Embedding shape: (${embedding.shape}), first value: ${embeddingData[0].toFixed(4)}`);
            // IMPORTANT: The 'embedding' tensor passed here is the CLONED/DETACHED one.
            // It's safe to use its data. It will be disposed when the object is removed
            // or a new embedding overwrites it in objectEmbeddingStatus.
        } catch (error) {
            console.error(`Error fetching/processing data from Vector DB for ID ${objectId}:`, error);
        }
    }

  // --- Helper to Draw Bounding Box WITH ID ---
  private drawBoundingBoxWithId(detectionData: number[], color: number[], trackId: number) {
    if (!this.ctx || !this.videoElement || !this.videoElement.videoWidth) { // Check videoElement exists
      return;
    }

    const confidence = detectionData[4];
    const videoBbox = this.scaleModelBboxToVideo(detectionData);
    if (!videoBbox) return;

    const canvasX1 = videoBbox.x1 * this.scale + this.offsetX;
    const canvasY1 = videoBbox.y1 * this.scale + this.offsetY;
    const canvasWidth = (videoBbox.x2 - videoBbox.x1) * this.scale;
    const canvasHeight = (videoBbox.y2 - videoBbox.y1) * this.scale;

    this.ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(canvasX1, canvasY1, canvasWidth, canvasHeight);

    this.ctx.font = '12px sans-serif';
    this.ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.8)`;
    const idLabel = trackId === -1 ? 'Init' : `ID: ${trackId}`;
    const label = `${idLabel} (${confidence.toFixed(2)})`;
    const textMetrics = this.ctx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = 12;
    const padding = 2;
    let textY = canvasY1 - padding - 1;
    let backgroundY = textY - textHeight - padding;
    if (backgroundY < 0) {
        textY = canvasY1 + textHeight + padding;
        backgroundY = canvasY1 + padding;
    }
    this.ctx.fillRect(canvasX1 - 1, backgroundY , textWidth + (padding * 2), textHeight + (padding * 2));
    this.ctx.fillStyle = `white`;
    this.ctx.fillText(label, canvasX1 + padding -1 , textY);
  }

  // Draws a single segmentation mask onto the offscreen mask canvas
  private async drawSegmentationMask(maskTensor: tf.Tensor2D, color: number[], bboxModel: number[]): Promise<void> {
    if (!this.maskCtx || maskTensor.isDisposed) {
        maskTensor?.dispose(); // Dispose if context missing or already disposed
        return;
    }

    try {
        const videoBbox = this.scaleModelBboxToVideo(bboxModel);
        if (!videoBbox) {
             maskTensor.dispose();
             return;
         }

        const [height, width] = maskTensor.shape;

        // Create colored mask data
        const colorTensor = tf.tensor1d(color.map(c => c / 255.0)); // Normalize color to [0,1]
        const maskColored = tf.tidy(() => {
            // Expand mask [H, W] -> [H, W, 1] and color [3] -> [1, 1, 3] for broadcasting
            return maskTensor.expandDims(-1).mul(colorTensor.reshape([1, 1, 3]));
        });

        // Convert tensor to ImageData for drawing
        // Add alpha channel (fully opaque for masked areas)
        const alpha = maskTensor.expandDims(-1); // Use the mask itself as alpha [H, W, 1]
        const maskRgba = tf.tidy(() => tf.concat([maskColored, alpha], -1)); // [H, W, 4]
        const maskPixelData = await tf.browser.toPixels(maskRgba);
        const imageData = new ImageData(maskPixelData, width, height);

        // Calculate position on the main canvas (using scaling and offset)
        const canvasX = videoBbox.x1 * this.scale + this.offsetX;
        const canvasY = videoBbox.y1 * this.scale + this.offsetY;

        // Draw the ImageData onto the offscreen mask canvas at the correct scaled position
        // We draw relative to the mask canvas origin (0,0) first, then draw the whole mask canvas later.
        // Need position relative to *video frame* on the mask canvas for correct placement
        // when the maskCanvas is drawn onto the main canvas.

        // Get top-left corner in *canvas* coordinates
        const canvasBoxX1 = videoBbox.x1 * this.scale + this.offsetX;
        const canvasBoxY1 = videoBbox.y1 * this.scale + this.offsetY;

        // Draw the mask onto the mask canvas
        this.maskCtx.putImageData(imageData, Math.round(canvasBoxX1), Math.round(canvasBoxY1));

        // Dispose tensors used in this function
        tf.dispose([maskTensor, colorTensor, maskColored, maskRgba, alpha]);

    } catch (error) {
        console.error("Error drawing segmentation mask:", error);
        if (!maskTensor.isDisposed) maskTensor.dispose(); // Ensure disposal on error
    }
  }

  // --- Render ---
  render() {
    let statusMessage = this.inferenceStatus;
    if (this.modelError || this.embedModelError) {
      statusMessage = `Error: ${this.modelError || this.embedModelError || 'Unknown model error'}`;
    } else if (this.isModelLoading || this.isEmbedModelLoading) {
      statusMessage = "Loading models...";
    } else if (!this.currentStream) {
      statusMessage = "Waiting for camera access...";
    } else if (!this.tfjsModel || !this.tfjsEmbedModel) {
      statusMessage = "Models loaded, waiting for stream...";
    } else if (this.videoElement && this.videoElement.readyState < this.videoElement.HAVE_METADATA) {
        statusMessage = "Waiting for video metadata...";
    } else if (this.inferenceLoopId === null) {
        statusMessage = "Ready to start inference.";
    }

    return html`
      <video id="video" muted playsinline></video>
      <canvas id="overlay-canvas"></canvas>
      <div class="status">${statusMessage}</div>
    `;
  }
}
