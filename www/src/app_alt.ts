// FULL CODE - ONLY drawSegmentationMask and related calculations are substantially changed

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
  bboxModel: number[]; // Full detection data [x1,y1,x2,y2,conf,class,...coeffs]
  point: Point;        // Center point [x,y] in model coords
  maskTensor?: tf.Tensor2D; // Optional mask tensor (binary, cropped to bbox size)
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
const PROTO_MASK_SIZE = 160; // Dimension (e.g., 160x160) of prototype masks
const MASK_COEFF_COUNT = 32; // Number of mask coefficients
// --- EMBED Configuration ---
const EMBED_INPUT_WIDTH = 128;
const EMBED_INPUT_HEIGHT = 192;
const EMBED_URL = '/assets/models/convnextv2_convlinear__aivb8jvk-47500__encoder__web_model/model.json';
const EMBEDDING_LOOP_INTERVAL_MS = 250; // How often to check for embedding tasks
const EMBEDDING_CROP_PADDING_FACTOR = 0.1; // Add 10% padding around bbox for crop
// --- Drawing Configuration ---
const MASK_DRAW_ALPHA = 0.5; // Alpha for drawing masks
// Set to 0 to disable padding for alignment testing. Re-introduce (e.g., 0.02) if alignment is good.
const MASK_CROP_PADDING_FACTOR = 0.0; // Padding for mask cropping (0% = use exact bbox)
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
      object-fit: contain; /* Maintain aspect ratio, letterbox/pillarbox */
      background-color: #000;
    }
    canvas#overlay-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%; /* Will be sized by JS */
      height: 100%; /* Will be sized by JS */
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
  @query('#video') private videoElement!: HTMLVideoElement;
  @query('#overlay-canvas') private canvasElement!: HTMLCanvasElement; // Visible canvas
  private ctx!: CanvasRenderingContext2D; // Context for visible canvas

  // --- Offscreen Buffer Canvas for Drawing ---
  private bufferCanvas!: HTMLCanvasElement;
  private bufferCtx!: CanvasRenderingContext2D;

  // --- TFJS State ---
  @state() private tfjsModel: tf.GraphModel | null = null;
  @state() private isModelLoading: boolean = true;
  @state() private modelError: string | null = null;
  @state() private tfjsEmbedModel: tf.GraphModel | null = null;
  @state() private isEmbedModelLoading: boolean = true;
  @state() private embedModelError: string | null = null;
  @state() private inferenceStatus: string = 'Initializing...';

  // --- Video Stream State ---
  private currentStream: MediaStream | null = null;
  private inferenceLoopId: number | null = null;
  private videoDisplayWidth: number = 0;
  private videoDisplayHeight: number = 0;
  private resizeObserver!: ResizeObserver;
  private isInferencing: boolean = false;
  private scale: number = 1; // Scale factor video -> display
  private offsetX: number = 0;// Offset for centering video on display
  private offsetY: number = 0;

  // --- Tracking state ---
  private tracker: Tracker | null = null;
  // --- Embedding State ---
  private objectEmbeddingStatus = new Map<number, ObjectEmbeddingInfo>();
  private embeddingLoopTimeoutId: number | null = null;
  private isEmbedding: boolean = false;

  // --- Lit Lifecycle ---

  connectedCallback() {
    super.connectedCallback();
    // --- Create Offscreen Buffer Canvas ---
    this.bufferCanvas = document.createElement('canvas');
    const bufCtx = this.bufferCanvas.getContext('2d', { willReadFrequently: true }); // willReadFrequently might be needed if we read back from it later
    if (!bufCtx) {
        this.handleFatalError("Failed to get 2D context for buffer canvas");
        return; // Stop initialization if buffer cannot be created
    }
    this.bufferCtx = bufCtx;
    // --- End Buffer Canvas Setup ---

    tf.ready().then(async () => {
      this.loadTfjsModels();
      this.startVideoStream();
      this.tracker = new Tracker({
          distanceThreshold: 100, // Adjust based on object speed and framerate
          hitInertiaMin: 0,
          hitInertiaMax: 15, // Allow objects to coast for ~15 frames
          initDelay: 2,     // Require 2 consecutive hits to initialize
      });
      console.log("Norfair tracker initialized.");
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.stopVideoStream();
    this.cancelEmbeddingLoop();
    this.tfjsModel?.dispose();
    this.tfjsEmbedModel?.dispose();
    this.tfjsModel = null;
    this.tfjsEmbedModel = null;
    this.resizeObserver?.disconnect();
    this.tracker = null;
    this.objectEmbeddingStatus.forEach(info => info.embedding?.dispose());
    this.objectEmbeddingStatus.clear();
    // Clean up buffer canvas if needed (usually handled by GC)
    // this.bufferCanvas = null; // Allow GC
    // this.bufferCtx = null; // Allow GC
  }

  protected firstUpdated(_changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>): void {
    if (!this.videoElement || !this.canvasElement) {
      this.handleFatalError("Initialization Error: Video or Canvas element missing.");
      return;
    }
    // Get context for the *visible* canvas
    const context = this.canvasElement.getContext('2d');
    if (!context) {
      this.handleFatalError("Visible canvas context error.");
      return;
    }
    this.ctx = context; // Assign to the main context property

    // Ensure buffer context was created successfully in connectedCallback
    if (!this.bufferCtx) {
        this.handleFatalError("Buffer canvas context not initialized.");
        return;
    }

    this.videoElement.addEventListener('loadedmetadata', this.handleVideoMetadataLoaded);
    this.videoElement.addEventListener('error', (e) => this.handleFatalError('Video element error.', e));

    // Observe the video element for size changes
    this.resizeObserver = new ResizeObserver(() => this.updateCanvasSize());
    this.resizeObserver.observe(this.videoElement);
    // Initial size update
    this.updateCanvasSize();
  }

  // --- Event Handlers ---
  private handleVideoMetadataLoaded = () => {
    this.videoElement.play().catch(e => this.handleFatalError('Video play failed:', e));
    this.updateCanvasSize(); // Update sizes now that metadata is loaded
    this.startInferenceLoop();
    this.startEmbeddingLoop();
  }

  private handleFatalError(message: string, error?: any) {
      console.error(message, error);
      this.inferenceStatus = message;
      this.stopVideoStream();
      this.cancelEmbeddingLoop();
      // Potentially clear canvases? Or leave last state?
      this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
      this.bufferCtx?.clearRect(0, 0, this.bufferCanvas?.width ?? 0, this.bufferCanvas?.height ?? 0);
      this.requestUpdate(); // Update status message display
  }

  // --- Size & Scaling ---
  private updateCanvasSize() {
    // Ensure elements and video dimensions are available
    if (!this.videoElement || !this.canvasElement || !this.bufferCanvas || !this.videoElement.videoWidth || this.videoElement.videoWidth === 0) {
      return;
    }
    // Get the actual displayed size of the video element
    const displayWidth = this.videoElement.clientWidth;
    const displayHeight = this.videoElement.clientHeight;

    // Check if dimensions have changed to avoid unnecessary resizing
    if (this.canvasElement.width !== displayWidth || this.canvasElement.height !== displayHeight) {
        // Resize the visible canvas
        this.canvasElement.width = displayWidth;
        this.canvasElement.height = displayHeight;

        // Resize the offscreen buffer canvas to match
        this.bufferCanvas.width = displayWidth;
        this.bufferCanvas.height = displayHeight;

        // Store the new dimensions
        this.videoDisplayWidth = displayWidth;
        this.videoDisplayHeight = displayHeight;
        console.log(`Canvas resized to: ${displayWidth}x${displayHeight}`);
    }

    // Calculate scale and offset to fit video within the display area
    const videoWidth = this.videoElement.videoWidth;
    const videoHeight = this.videoElement.videoHeight;
    const scaleX = this.videoDisplayWidth / videoWidth;
    const scaleY = this.videoDisplayHeight / videoHeight;
    this.scale = Math.min(scaleX, scaleY); // Use 'contain' scaling
    // Calculate offsets to center the scaled video
    this.offsetX = (this.videoDisplayWidth - videoWidth * this.scale) / 2;
    this.offsetY = (this.videoDisplayHeight - videoHeight * this.scale) / 2;
  }


  // --- TFJS Methods ---
  private async loadTfjsModels() {
    // ... (No changes) ...
    this.isModelLoading = true; this.isEmbedModelLoading = true;
    this.modelError = null; this.embedModelError = null;
    this.inferenceStatus = 'Loading models...'; this.requestUpdate();

    try {
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
    // ... (No changes) ...
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
    // ... (No changes) ...
    this.cancelInferenceLoop();
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop());
    }
    this.currentStream = null;
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.srcObject = null;
    }
     // Clear both canvases when stream stops
    this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
    this.bufferCtx?.clearRect(0, 0, this.bufferCanvas?.width ?? 0, this.bufferCanvas?.height ?? 0);
    if (!this.modelError && !this.embedModelError) {
        this.inferenceStatus = "Video stream stopped.";
    }
    this.requestUpdate();
  }

  // --- Inference Loop (YOLO + Tracking) ---
  private startInferenceLoop() {
    // ... (No changes) ...
    if (this.inferenceLoopId !== null || this.isModelLoading || !this.currentStream || !this.tfjsModel || !this.videoElement || this.videoElement.readyState < this.videoElement.HAVE_METADATA) {
      return;
    }
    console.log("Starting YOLO/Tracking inference loop...");
    this.isInferencing = false;
    // Use requestAnimationFrame for smooth looping synchronized with display refresh
    this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
  }

  private cancelInferenceLoop() {
    // ... (No changes) ...
     if (this.inferenceLoopId !== null) {
      cancelAnimationFrame(this.inferenceLoopId);
      this.inferenceLoopId = null;
      this.isInferencing = false;
      if (!this.modelError && !this.embedModelError) {
        this.inferenceStatus = "Inference stopped.";
         // Clear canvases when loop stops
        this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
        this.bufferCtx?.clearRect(0, 0, this.bufferCanvas?.width ?? 0, this.bufferCanvas?.height ?? 0);
        this.requestUpdate();
      }
      console.log("YOLO/Tracking inference loop cancelled.");
    }
  }

  private async runInference() {
    if (this.isInferencing) { return; } // Prevent re-entry
    // Check prerequisites
    if (!this.tfjsModel || !this.currentStream || !this.videoElement || !this.ctx || !this.bufferCtx || this.videoElement.paused || this.videoElement.ended || this.videoElement.readyState < this.videoElement.HAVE_CURRENT_DATA || !this.tracker) {
        // If prerequisites not met, request next frame and exit
        if (this.inferenceLoopId !== null) { // Avoid scheduling if cancelled
             this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
        }
        return;
    }

    this.isInferencing = true; // Mark as busy
    this.updateCanvasSize(); // Ensure canvas sizes are correct

    // --- Clear only the offscreen buffer canvas at the start ---
    this.bufferCtx.clearRect(0, 0, this.bufferCanvas.width, this.bufferCanvas.height);

    const tensorsToDispose: tf.Tensor[] = [];
    let output: tf.Tensor[] | null = null;
    const frameTime = Date.now(); // For object status tracking

    try {
        // 1. Get video frame and preprocess
        const frameTensor = tf.browser.fromPixels(this.videoElement);
        tensorsToDispose.push(frameTensor);

        const inputTensor = tf.tidy(() => {
            const resized = tf.image.resizeBilinear(frameTensor, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]);
            const normalized = resized.div(255.0);
            return normalized.expandDims(0).cast('float32'); // Add batch dim
        });
        // inputTensor is managed by tidy, but frameTensor needs manual disposal

        // 2. Run YOLO model
        output = await this.tfjsModel!.executeAsync(inputTensor) as tf.Tensor[];
        if (output) {
          tensorsToDispose.push(...output); // Add model outputs for disposal
        }

        // 3. Process Detections and Masks
        if (output && output.length >= 3) {
            const detectionsTensor = output[0]; // [1, num_dets, 38]
            const protoTensor = output[2];      // [1, 160, 160, 32]

            // --- (Shape validation remains the same) ---
             if (protoTensor.shape.length !== 4 || protoTensor.shape[0] !== 1 ||
                 protoTensor.shape[1] !== PROTO_MASK_SIZE || protoTensor.shape[2] !== PROTO_MASK_SIZE ||
                 protoTensor.shape[3] !== MASK_COEFF_COUNT) {
                  console.error("Unexpected prototype mask tensor shape:", protoTensor.shape);
                  throw new Error("Incorrect prototype mask shape.");
             }

            const detectionsBatch = (await detectionsTensor.data()) as Float32Array;
            const numDets = detectionsTensor.shape[1];
            const detDataLength = detectionsTensor.shape[2];

            const norfairDetections: Point[] = [];
            const currentFrameDetectionResults = new Map<number, DetectionResult>();

            // --- Loop through raw detections ---
            for (let i = 0; i < numDets; i++) {
                const offset = i * detDataLength;
                const confidence = detectionsBatch[offset + 4];

                if (confidence >= CONFIDENCE_THRESHOLD) {
                    const modelX1 = detectionsBatch[offset + 0];
                    const modelY1 = detectionsBatch[offset + 1];
                    const modelX2 = detectionsBatch[offset + 2];
                    const modelY2 = detectionsBatch[offset + 3];
                    const maskCoeffs = tf.slice(detectionsTensor, [0, i, 6], [1, 1, MASK_COEFF_COUNT]); // [1, 1, 32]

                    const centerX = (modelX1 + modelX2) / 2;
                    const centerY = (modelY1 + modelY2) / 2;
                    const point: Point = [centerX, centerY]; // For Norfair tracker
                    const norfairIndex = norfairDetections.length;
                    norfairDetections.push(point);

                    // --- Calculate Mask (inside tidy scope) ---
                    const binaryMask = tf.tidy(() => {
                        // Combine protos and coeffs (matmul)
                        const protosReshaped = protoTensor.squeeze(0).reshape([PROTO_MASK_SIZE * PROTO_MASK_SIZE, MASK_COEFF_COUNT]); // [25600, 32]
                        const coeffsReshaped = maskCoeffs.reshape([MASK_COEFF_COUNT, 1]); // [32, 1]
                        const maskProto = tf.matMul(protosReshaped, coeffsReshaped); // [25600, 1]
                        const maskReshaped = maskProto.reshape([PROTO_MASK_SIZE, PROTO_MASK_SIZE]); // [160, 160]
                        const maskActivated = tf.sigmoid(maskReshaped);

                        // Get BBox in video coordinates (unpadded)
                        const videoBbox = this.scaleModelBboxToVideo([modelX1, modelY1, modelX2, modelY2]);
                        if (!videoBbox) return null; // Should not happen if video dimensions known

                        const videoWidth = this.videoElement.videoWidth;
                        const videoHeight = this.videoElement.videoHeight;

                        // --- Calculate Crop Box (using MASK_CROP_PADDING_FACTOR=0.0 for now) ---
                        const boxWidth = videoBbox.x2 - videoBbox.x1;
                        const boxHeight = videoBbox.y2 - videoBbox.y1;
                        const padX = boxWidth * MASK_CROP_PADDING_FACTOR;
                        const padY = boxHeight * MASK_CROP_PADDING_FACTOR;
                        // Padded video coordinates for cropping the mask (currently same as unpadded)
                        const cropX1 = Math.max(0, videoBbox.x1 - padX);
                        const cropY1 = Math.max(0, videoBbox.y1 - padY);
                        const cropX2 = Math.min(videoWidth, videoBbox.x2 + padX);
                        const cropY2 = Math.min(videoHeight, videoBbox.y2 + padY);
                        // Target width/height for the cropped mask
                        const targetCropW = Math.ceil(cropX2 - cropX1);
                        const targetCropH = Math.ceil(cropY2 - cropY1);
                        // ---

                        if (targetCropW <= 0 || targetCropH <= 0) return null; // Invalid crop dimensions

                        // Normalize the *padded* box for cropAndResize input
                        const normalizedPaddedBbox = [[
                            cropY1 / videoHeight, cropX1 / videoWidth,
                            cropY2 / videoHeight, cropX2 / videoWidth
                        ]];

                        const maskExpanded = maskActivated.expandDims(0).expandDims(-1); // [1, 160, 160, 1]

                        // Crop and resize the activated mask to the target (padded) dimensions
                        const maskCroppedResized = tf.image.cropAndResize(
                            maskExpanded,
                            normalizedPaddedBbox,
                            [0], // box index
                            [targetCropH, targetCropW], // target size
                            'bilinear' // interpolation method
                        );

                        // Threshold and finalize mask shape
                        const finalMask = maskCroppedResized.squeeze([0, 3]).greater(MASK_THRESHOLD).cast('float32'); // [targetCropH, targetCropW]

                        return finalMask as tf.Tensor2D;
                    }); // End mask tf.tidy

                    // Store results needed for drawing and tracking
                    currentFrameDetectionResults.set(norfairIndex, {
                        bboxModel: Array.from(detectionsBatch.slice(offset, offset + detDataLength)), // Store raw model output
                        point: point,
                        maskTensor: binaryMask || undefined // Store the final binary mask tensor
                    });

                    maskCoeffs.dispose(); // Dispose intermediate coeff tensor

                } // End confidence check
            } // End detection loop

            // 4. Update Tracker
            const trackingResults = this.tracker.update(norfairDetections);
            this.updateObjectStatus(this.tracker.trackedObjects, frameTime);

            // 5. Drawing (using Double Buffering)
            let trackedDetectionCount = 0;
            const maskPromises: Promise<void>[] = []; // To await async mask drawing

            // --- Draw BBoxes synchronously onto buffer ---
            for (let i = 0; i < norfairDetections.length; i++) {
                 const assignedId = trackingResults[i];
                 const detResult = currentFrameDetectionResults.get(i);
                 if (detResult) {
                     trackedDetectionCount++;
                     const colorIndex = assignedId >= 0 ? assignedId % COLORS.length : COLORS.length - 1;
                     const color = COLORS[colorIndex];

                     // Draw BBox onto the BUFFER canvas
                     this.drawBoundingBoxWithId(this.bufferCtx, detResult.bboxModel, color, assignedId);

                     // Prepare mask drawing promise (will draw onto BUFFER canvas)
                     if (detResult.maskTensor) {
                          // Pass buffer context, mask tensor, color, and the *original model bbox*
                          maskPromises.push(this.drawSegmentationMask(this.bufferCtx, detResult.maskTensor, color, detResult.bboxModel.slice(0, 4)));
                     }

                     // Update embedding status (uses video bbox directly)
                     if (assignedId !== -1 && this.objectEmbeddingStatus.has(assignedId)) {
                          const videoBbox = this.scaleModelBboxToVideo(detResult.bboxModel.slice(0,4)); // Use only bbox part
                          if (videoBbox) {
                               this.objectEmbeddingStatus.get(assignedId)!.lastKnownBboxVideo = videoBbox;
                          }
                     }
                 } else {
                      // Clean up mask tensor if detection somehow lost
                      currentFrameDetectionResults.get(i)?.maskTensor?.dispose();
                 }
            }

            // --- Wait for all async mask drawings onto the buffer to complete ---
            await Promise.all(maskPromises);

            // --- Draw the complete buffer onto the visible canvas ---
            this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height); // Clear visible canvas
            this.ctx.drawImage(this.bufferCanvas, 0, 0); // Draw buffer content

            this.inferenceStatus = `Tracked Detections: ${trackedDetectionCount}`;
            // --- End Double Buffer Drawing ---

        } else { // Handle case where model output is missing tensors
             console.warn("YOLO output missing expected tensors:", output?.length);
             this.inferenceStatus = "Model output missing tensors.";
             this.tracker.update([]); // Update tracker with no detections
             this.updateObjectStatus(this.tracker.trackedObjects, frameTime);
             // Clear visible canvas directly if buffer wasn't drawn
             this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        }

    } catch (error: any) {
        console.error("Error during YOLO/Tracking/Masking inference:", error);
        this.inferenceStatus = `Inference Error: ${error.message || error}`;
        // Don't draw buffer on error, just clear visible canvas
        this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
        if (this.tracker) { this.tracker.update([]); this.updateObjectStatus(this.tracker.trackedObjects, frameTime); }
    } finally {
        tf.dispose(tensorsToDispose); // Dispose manually tracked tensors
        this.isInferencing = false; // Release lock
        this.requestUpdate(); // Update status display
        // Schedule the next frame
        if (this.inferenceLoopId !== null) {
             this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
        }
    }
  }


   // --- Embedding Loop ---
   private startEmbeddingLoop() {
    // ... (No changes) ...
     if (this.embeddingLoopTimeoutId !== null || this.isEmbedModelLoading || !this.currentStream || !this.tfjsEmbedModel || !this.videoElement || this.videoElement.readyState < this.videoElement.HAVE_METADATA) {
        return;
    }
    console.log("Starting Embedding loop...");
    this.isEmbedding = false;
    this.embeddingLoopTimeoutId = window.setTimeout(() => this.runEmbeddingLoop(), EMBEDDING_LOOP_INTERVAL_MS);
   }

   private cancelEmbeddingLoop() {
    // ... (No changes) ...
     if (this.embeddingLoopTimeoutId !== null) {
        clearTimeout(this.embeddingLoopTimeoutId);
        this.embeddingLoopTimeoutId = null;
        this.isEmbedding = false;
        console.log("Embedding loop cancelled.");
    }
   }

    private async runEmbeddingLoop() {
       // ... (No changes) ...
        if (this.isEmbedding) {
            // console.log("Already embedding, skipping cycle."); // Less verbose
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
                 const bboxVideo = objectInfo.lastKnownBboxVideo; // Uses the stored video bbox

                 if (bboxVideo) {
                    // console.log(`Embedding object ID: ${objectIdForEmbedding}`); // Less verbose

                    // --- Manual Tensor Management Scope ---
                    let imageTensor: tf.Tensor | null = null;
                    let cropped: tf.Tensor | null = null;
                    let normalized: tf.Tensor | null = null;
                    let embedding: tf.Tensor | null = null;
                    const intermediateTensors: tf.Tensor[] = []; // Track intermediate tensors

                    try {
                        // 1. Crop Image from Video Frame using padded bbox
                        imageTensor = tf.browser.fromPixels(this.videoElement);
                        intermediateTensors.push(imageTensor);

                        const videoWidth = this.videoElement.videoWidth;
                        const videoHeight = this.videoElement.videoHeight;
                        const boxWidth = bboxVideo.x2 - bboxVideo.x1;
                        const boxHeight = bboxVideo.y2 - bboxVideo.y1;
                        // Use EMBEDDING padding factor here
                        const padX = boxWidth * EMBEDDING_CROP_PADDING_FACTOR;
                        const padY = boxHeight * EMBEDDING_CROP_PADDING_FACTOR;
                        const cropX1 = Math.max(0, Math.floor(bboxVideo.x1 - padX));
                        const cropY1 = Math.max(0, Math.floor(bboxVideo.y1 - padY));
                        const cropX2 = Math.min(videoWidth, Math.ceil(bboxVideo.x2 + padX));
                        const cropY2 = Math.min(videoHeight, Math.ceil(bboxVideo.y2 + padY));

                        if (cropX2 <= cropX1 || cropY2 <= cropY1) {
                            // Don't throw error, just skip embedding for this frame
                            console.warn(`Invalid crop dimensions for embedding object ${objectIdForEmbedding}, skipping.`);
                            return; // Exit the try block for this object
                        }

                        const boxes = [[ cropY1 / videoHeight, cropX1 / videoWidth, cropY2 / videoHeight, cropX2 / videoWidth ]];
                        const boxIndices = [0];
                        const cropSize: [number, number] = [EMBED_INPUT_HEIGHT, EMBED_INPUT_WIDTH];

                        // Wrap crop/normalize/execute in tidy for better memory management? Maybe not needed with manual scope.
                        cropped = tf.image.cropAndResize(
                            imageTensor.expandDims(0).toFloat(), // Add batch dim
                            boxes, boxIndices, cropSize, 'bilinear'
                        );
                        intermediateTensors.push(cropped);

                        // 2. Normalize
                        normalized = cropped.div(255.0);
                        // Let executeAsync handle the normalized tensor's memory if possible
                        // intermediateTensors.push(normalized); // Only if not input to execute

                        // 3. Run Embedding Model
                        embedding = await this.tfjsEmbedModel!.executeAsync(normalized) as tf.Tensor;


                        if (embedding && embedding.shape && embedding.shape.length > 0) { // Check if valid tensor
                            // Clone result BEFORE disposing intermediates/inputs
                            finalEmbeddingTensor = embedding.clone();
                        } else {
                             console.warn(`Embedding model execution returned invalid tensor for object ${objectIdForEmbedding}`);
                             // embedding?.dispose(); // Dispose if invalid but exists
                             return; // Skip storing/fetching
                        }

                    } finally {
                        // Dispose intermediate tensors manually
                        tf.dispose(intermediateTensors);
                        // Dispose the raw embedding output if it wasn't the final tensor (already cloned)
                        if (embedding && !embedding.isDisposed && embedding !== finalEmbeddingTensor) {
                             embedding.dispose();
                        }
                        // Dispose normalized tensor if it wasn't passed to executeAsync and wasn't disposed
                        if (normalized && !normalized.isDisposed && normalized !== embedding) {
                             normalized.dispose();
                        }
                    }
                    // --- End Manual Tensor Management Scope ---

                    if (finalEmbeddingTensor) {
                         objectInfo.embedding?.dispose(); // Dispose previous embedding
                         objectInfo.embedding = finalEmbeddingTensor; // Store the new clone
                         objectInfo.hasBeenEmbedded = true;
                         objectInfo.lastEmbeddingTime = Date.now();
                         // console.log(`Stored new embedding for object ID: ${objectIdForEmbedding}, Shape: ${finalEmbeddingTensor.shape}`);

                         await this.fetchDataFromVectorDB(finalEmbeddingTensor, objectIdForEmbedding);
                    }
                 } else {
                     // console.log(`Skipping embedding for ID ${objectIdForEmbedding}, no bbox found.`); // Less verbose
                 }
            } else {
                 // console.log("No suitable object found for embedding this cycle."); // Less verbose
            }

        } catch (error: any) {
            console.error(`Error during embedding loop for object ${objectIdForEmbedding}:`, error);
            // Clean up the final tensor if an error occurred after creation
            if (finalEmbeddingTensor && !finalEmbeddingTensor.isDisposed){
                 finalEmbeddingTensor.dispose();
            }
        } finally {
            this.isEmbedding = false;
            if (this.embeddingLoopTimeoutId !== null) {
                // Schedule next check
                this.embeddingLoopTimeoutId = window.setTimeout(() => this.runEmbeddingLoop(), EMBEDDING_LOOP_INTERVAL_MS);
            }
        }
    }

    // --- Helper to select the next object for embedding based on priority ---
    private selectObjectForEmbedding(): TrackedObject | null {
        // ... (No changes) ...
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

        // Fallback: if only embedded objects with no timestamp exist (shouldn't happen often)
        const fallback = candidatesInfo.filter(info => info.hasBeenEmbedded);
        if(fallback.length > 0) {
             return activeObjects.find(obj => obj.id === fallback[0].id) || null;
        }


        return null;
    }

    // --- Helper to manage object status map ---
    private updateObjectStatus(currentTrackedObjects: TrackedObject[], timestamp: number) {
        // ... (No changes) ...
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
            // console.log(`Removing object ID ${id} from embedding status.`); // Less verbose
            info?.embedding?.dispose(); // Dispose stored embedding
            this.objectEmbeddingStatus.delete(id);
        });

        // Add newly initialized objects
        currentTrackedObjects.forEach(obj => {
            if (obj.id !== -1 && !this.objectEmbeddingStatus.has(obj.id)) {
                // console.log(`Adding object ID ${obj.id} to embedding status.`); // Less verbose
                this.objectEmbeddingStatus.set(obj.id, {
                    id: obj.id,
                    hasBeenEmbedded: false,
                    lastEmbeddingTime: null,
                    creationTime: timestamp,
                    embedding: null,
                    lastKnownBboxVideo: null, // Will be updated in inference loop
                });
            }
        });
    }

    // --- Helper to scale model bbox coordinates to video coordinates ---
    private scaleModelBboxToVideo(bboxModel: number[]): { x1: number, y1: number, x2: number, y2: number } | null {
        // ... (No changes) ...
        if (!this.videoElement?.videoWidth) return null; // Check videoElement exists and has width
        const videoWidth = this.videoElement.videoWidth;
        const videoHeight = this.videoElement.videoHeight;
        // Ensure we have 4 elements
        if(bboxModel.length < 4) return null;
        const [x1_m, y1_m, x2_m, y2_m] = bboxModel;

        // Scale coordinates from model input size to native video size
        const videoX1 = (x1_m / MODEL_INPUT_WIDTH) * videoWidth;
        const videoY1 = (y1_m / MODEL_INPUT_HEIGHT) * videoHeight;
        const videoX2 = (x2_m / MODEL_INPUT_WIDTH) * videoWidth;
        const videoY2 = (y2_m / MODEL_INPUT_HEIGHT) * videoHeight;

        return { x1: videoX1, y1: videoY1, x2: videoX2, y2: videoY2 };
    }

    // --- Placeholder for Vector DB Query ---
    private async fetchDataFromVectorDB(embedding: tf.Tensor, objectId: number): Promise<void> {
        // ... (No changes) ...
        // console.log(`Workspaceing data for object ID ${objectId} using embedding...`); // Less verbose
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 50));

        try {
            // Example: Convert tensor data to array for sending (if needed)
            // const embeddingData = await embedding.data();
            // console.log(` -> Simulated DB query complete for ID ${objectId}. Embedding shape: (${embedding.shape}), first value: ${embeddingData[0].toFixed(4)}`);
             // console.log(` -> Simulated DB query complete for ID ${objectId}. Embedding shape: (${embedding.shape})`); // Less verbose
            // IMPORTANT: The 'embedding' tensor passed here is the CLONED/DETACHED one.
            // It's safe to use its data. It will be disposed when the object is removed
            // or a new embedding overwrites it in objectEmbeddingStatus.
        } catch (error) {
            console.error(`Error fetching/processing data from Vector DB for ID ${objectId}:`, error);
        }
    }

  // --- Helper to Draw Bounding Box WITH ID onto a given context ---
  private drawBoundingBoxWithId(drawCtx: CanvasRenderingContext2D, detectionData: number[], color: number[], trackId: number) {
    // ... (No changes) ...
     if (!drawCtx || !this.videoElement?.videoWidth) {
      return;
    }

    // Use the first 4 elements for the bbox
    const modelBbox = detectionData.slice(0, 4);
    const confidence = detectionData[4]; // Assuming conf is 5th element
    // const classId = detectionData[5]; // Assuming class is 6th element

    // Scale the MODEL bbox coordinates to VIDEO coordinates
    const videoBbox = this.scaleModelBboxToVideo(modelBbox);
    if (!videoBbox) return; // Skip if scaling failed

    // Scale the VIDEO coordinates to DISPLAY CANVAS coordinates
    const canvasX1 = videoBbox.x1 * this.scale + this.offsetX;
    const canvasY1 = videoBbox.y1 * this.scale + this.offsetY;
    const canvasWidth = (videoBbox.x2 - videoBbox.x1) * this.scale;
    const canvasHeight = (videoBbox.y2 - videoBbox.y1) * this.scale;

    // --- Draw on the provided context (e.g., bufferCtx) ---
    drawCtx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    drawCtx.lineWidth = 2;
    drawCtx.strokeRect(canvasX1, canvasY1, canvasWidth, canvasHeight);

    // Draw ID and confidence label
    drawCtx.font = '12px sans-serif';
    drawCtx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.8)`; // Semi-transparent background
    const idLabel = trackId === -1 ? 'Init' : `ID: ${trackId}`;
    const label = `${idLabel} (${confidence.toFixed(2)})`;
    const textMetrics = drawCtx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = 12; // Approximate height based on font size
    const padding = 2;

    // Position label above the box, moving inside if near top edge
    let textY = canvasY1 - padding - 1; // Text baseline position
    let backgroundY = textY - textHeight - padding; // Background rect top Y
    if (backgroundY < 0) { // If background goes offscreen top
        textY = canvasY1 + canvasHeight + textHeight + padding; // Place below box
        backgroundY = canvasY1 + canvasHeight + padding;
    }
    // Draw background rectangle
    drawCtx.fillRect(canvasX1 - 1, backgroundY , textWidth + (padding * 2), textHeight + (padding * 2));
    // Draw text
    drawCtx.fillStyle = `white`; // White text
    drawCtx.fillText(label, canvasX1 + padding -1 , textY);
    // --- End Drawing ---
  }

  // --- Draws a single segmentation mask onto the given context using drawImage for scaling ---
  private async drawSegmentationMask(drawCtx: CanvasRenderingContext2D, maskTensor: tf.Tensor2D, color: number[], modelBbox: number[]): Promise<void> {
    // *** MODIFIED FUNCTION ***
    if (!drawCtx || maskTensor.isDisposed || !this.videoElement?.videoWidth) {
        maskTensor?.dispose();
        return;
    }

    let tempMaskCanvas: HTMLCanvasElement | null = null; // For intermediate drawing
    const tensorsToDisposeInternally: tf.Tensor[] = [maskTensor]; // Track tensors created here + input

    try {
        // 1. Get the UNPADDED video bounding box coordinates (for positioning)
        const videoBbox = this.scaleModelBboxToVideo(modelBbox); // modelBbox is already [x1,y1,x2,y2]
        if (!videoBbox) { throw new Error("Failed to scale model bbox"); }

        // 2. Calculate VIDEO coordinates for the crop area (using padding factor)
        //    Needed to know the *source* dimensions of the maskTensor
        const videoWidth = this.videoElement.videoWidth;
        const videoHeight = this.videoElement.videoHeight;
        const boxWidth = videoBbox.x2 - videoBbox.x1;
        const boxHeight = videoBbox.y2 - videoBbox.y1;
        const padX = boxWidth * MASK_CROP_PADDING_FACTOR;
        const padY = boxHeight * MASK_CROP_PADDING_FACTOR;
        const cropX1 = Math.max(0, videoBbox.x1 - padX);
        const cropY1 = Math.max(0, videoBbox.y1 - padY);
        const cropX2 = Math.min(videoWidth, videoBbox.x2 + padX);
        const cropY2 = Math.min(videoHeight, videoBbox.y2 + padY);
        // These are the dimensions the maskTensor corresponds to in video space
        const maskVideoWidth = cropX2 - cropX1;
        const maskVideoHeight = cropY2 - cropY1;

        if (maskVideoWidth <= 0 || maskVideoHeight <= 0) {
            throw new Error("Invalid mask video dimensions after padding/clipping");
        }

        // 3. Get mask dimensions (shape of the input tensor after cropAndResize)
        const [maskTensorHeight, maskTensorWidth] = maskTensor.shape; // e.g., [targetCropH, targetCropW]

        // 4. Convert mask tensor to colored ImageData
        const colorTensor = tf.tensor1d(color.map(c => c / 255.0));
        tensorsToDisposeInternally.push(colorTensor);
        const maskColored = tf.tidy(() => maskTensor.expandDims(-1).mul(colorTensor.reshape([1, 1, 3])));
        tensorsToDisposeInternally.push(maskColored);
        const alphaChannel = maskTensor.expandDims(-1);
        tensorsToDisposeInternally.push(alphaChannel);
        const maskRgba = tf.tidy(() => tf.concat([maskColored, alphaChannel], -1));
        tensorsToDisposeInternally.push(maskRgba); // Ensure maskRgba is disposed

        const maskPixelData = await tf.browser.toPixels(maskRgba); // Async step!
        const imageData = new ImageData(maskPixelData, maskTensorWidth, maskTensorHeight);

        // 5. Create a temporary canvas and put the ImageData onto it
        tempMaskCanvas = document.createElement('canvas');
        tempMaskCanvas.width = maskTensorWidth;
        tempMaskCanvas.height = maskTensorHeight;
        const tempCtx = tempMaskCanvas.getContext('2d');
        if (!tempCtx) { throw new Error("Failed to get context for temp mask canvas"); }
        tempCtx.putImageData(imageData, 0, 0);

        // 6. Calculate the destination rectangle on the DRAW CANVAS (e.g., bufferCtx)
        //    Position: Top-left corner of the CROP area, scaled to canvas.
        const canvasDrawX = cropX1 * this.scale + this.offsetX;
        const canvasDrawY = cropY1 * this.scale + this.offsetY;
        //    Size: Dimensions of the mask in VIDEO space, scaled to canvas.
        const canvasMaskWidth = maskVideoWidth * this.scale;
        const canvasMaskHeight = maskVideoHeight * this.scale;

        // 7. Draw the temporary canvas onto the destination context, SCALING it.
        drawCtx.save();
        drawCtx.globalAlpha = MASK_DRAW_ALPHA; // Apply transparency
        // Draw the temp canvas (which holds the mask) onto the buffer, scaling it to fit the calculated canvas dimensions
        drawCtx.drawImage(tempMaskCanvas, canvasDrawX, canvasDrawY, canvasMaskWidth, canvasMaskHeight);
        drawCtx.restore();

    } catch (error) {
        console.error("Error drawing segmentation mask:", error);
        // Ensure disposal on error if tensor still exists (handled below)
    } finally {
        // 8. Dispose temporary TF tensors created within this function scope
        tf.dispose(tensorsToDisposeInternally);
        // Temporary canvas (tempMaskCanvas) will be garbage collected.
    }
  }


  // --- Render ---
  render() {
    // Determine status message based on component state
    let statusMessage = this.inferenceStatus;
    if (this.modelError || this.embedModelError) {
      statusMessage = `Error: ${this.modelError || this.embedModelError || 'Unknown model error'}`;
    } else if (this.isModelLoading || this.isEmbedModelLoading) {
      statusMessage = "Loading models...";
    } else if (!this.currentStream) {
      statusMessage = "Waiting for camera access...";
    } else if (!this.tfjsModel || !this.tfjsEmbedModel) {
      // This state might not be reachable if loading handles errors correctly
      statusMessage = "Models loaded, waiting for stream...";
    } else if (this.videoElement && this.videoElement.readyState < this.videoElement.HAVE_METADATA) {
        statusMessage = "Waiting for video metadata...";
    } else if (this.inferenceLoopId === null && !this.isModelLoading && !this.isEmbedModelLoading) {
        // Only show ready if models are loaded and loop isn't running (e.g., after stop)
        statusMessage = "Ready."; // Or "Stopped."
    }

    return html`
      <video id="video" muted playsinline></video>
      <canvas id="overlay-canvas"></canvas> <div class="status">${statusMessage}</div>
    `;
  }
} // End VideoContainer class
