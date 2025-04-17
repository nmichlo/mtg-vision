import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, state, query, customElement } from 'lit/decorators.js';
import * as tf from '@tensorflow/tfjs';

import Tracker from "./norfair"

// --- Example Usage (Conceptual) ---
/*
const tracker = new Tracker({ distanceThreshold: 50 });

// Example frame detections (replace with actual data)
const frame1Detections: Point[] = [[100, 100], [300, 300]];
const frame2Detections: Point[] = [[105, 105], [350, 350]]; // obj1 moved, obj2 disappeared, new obj3 appears

const results1 = tracker.update(frame1Detections);
console.log("Frame 1 Results:", results1); // Likely [-1, -1] or [0, 1] after init delay

const results2 = tracker.update(frame2Detections);
console.log("Frame 2 Results:", results2); // Should show mapping for obj1, maybe -1 for the new one

// Access tracked objects (e.g., after initialization)
tracker.trackedObjects.forEach(obj => {
    if (obj.id !== -1) {
        console.log(`Object ID: ${obj.id}, Estimate: ${obj.estimate()}`);
    }
});
*/

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

// ===========================================================
// Video Container Component
// ===========================================================

// --- Configuration ---
const MODEL_URL = '/assets/models/yolov11s_seg__dk964hap__web_model/model.json';
const MODEL_INPUT_WIDTH = 640;
const MODEL_INPUT_HEIGHT = 640;
const CONFIDENCE_THRESHOLD = 0.5; // Filter detections below this score
// --- Color Palette ---
const COLORS = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    [255, 0, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0]
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
  @state() private tfjsModel: tf.GraphModel | null = null;
  @state() private isModelLoading: boolean = true;
  @state() private modelError: string | null = null;
  @state() private inferenceStatus: string = 'Initializing...';

  // --- Video Stream State ---
  private currentStream: MediaStream | null = null;
  private inferenceLoopId: number | null = null;
  private videoDisplayWidth: number = 0;
  private videoDisplayHeight: number = 0;
  private resizeObserver!: ResizeObserver;
  private isInferencing: boolean = false;
  private scale: number = 1;  // Store scale factors for drawing
  private offsetX: number = 0;
  private offsetY: number = 0;

  // --- Tracking state ---


  // --- Lit Lifecycle ---

  connectedCallback() {
    super.connectedCallback();
    tf.ready().then(() => {
      this.loadTfjsModel();
      this.startVideoStream();
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.stopVideoStream();
    this.tfjsModel?.dispose();
    this.tfjsModel = null;
    this.resizeObserver?.disconnect();
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
  }

  private handleFatalError(message: string, error?: any) {
      console.error(message, error);
      this.inferenceStatus = message;
      this.stopVideoStream(); // Stop everything
      this.requestUpdate();
  }

  // --- Size & Scaling ---
  private updateCanvasSize() {
    if (!this.videoElement || !this.canvasElement) return;
    const displayWidth = this.videoElement.clientWidth;
    const displayHeight = this.videoElement.clientHeight;

    // Only resize if needed
    if (this.canvasElement.width !== displayWidth || this.canvasElement.height !== displayHeight) {
      this.canvasElement.width = displayWidth;
      this.canvasElement.height = displayHeight;
      this.videoDisplayWidth = displayWidth;
      this.videoDisplayHeight = displayHeight;
      console.log(`Canvas resized to: ${displayWidth}x${displayHeight}`);
    }

    // Recalculate scaling factors whenever canvas might have resized
    const videoWidth = this.videoElement.videoWidth || MODEL_INPUT_WIDTH; // Fallback if video not ready
    const videoHeight = this.videoElement.videoHeight || MODEL_INPUT_HEIGHT;
    const scaleX = this.videoDisplayWidth / videoWidth;
    const scaleY = this.videoDisplayHeight / videoHeight;
    this.scale = Math.min(scaleX, scaleY); // Use minimum scale for 'contain'
    this.offsetX = (this.videoDisplayWidth - videoWidth * this.scale) / 2;
    this.offsetY = (this.videoDisplayHeight - videoHeight * this.scale) / 2;
  }

  // --- TFJS Methods ---
  private async loadTfjsModel() {
    this.isModelLoading = true; this.modelError = null; this.inferenceStatus = 'Loading model...'; this.requestUpdate();
    try {
      this.tfjsModel = await tf.loadGraphModel(MODEL_URL);
      this.isModelLoading = false;
      this.inferenceStatus = 'Model loaded. Waiting for video...';
      console.log('TFJS model loaded successfully.');
      this.startInferenceLoop();
    } catch (error: any) {
      this.isModelLoading = false;
      this.modelError = `Load failed: ${error.message || error}`;
      this.inferenceStatus = this.modelError;
      console.error('TFJS model load failed:', error);
    } finally {
      this.requestUpdate();
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
        this.inferenceStatus = this.isModelLoading ? 'Model loading...' : 'Video stream started.';
      } else { this.inferenceStatus = 'Video element not ready.'; }
    } catch (error: any) {
      this.inferenceStatus = `Camera failed: ${error.message}`;
      console.error('getUserMedia failed:', error);
      this.currentStream = null;
      if (this.videoElement) {
        this.videoElement.srcObject = null;
      }
    } finally { this.requestUpdate(); }
  }

  private stopVideoStream() {
    this.cancelInferenceLoop();
    if (this.currentStream) { this.currentStream.getTracks().forEach(track => track.stop()); }
    this.currentStream = null;
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.srcObject = null;
    }
    this.inferenceStatus = "Video stream stopped.";
    this.requestUpdate();
  }

  // --- Inference Loop ---
  private startInferenceLoop() {
    if (this.inferenceLoopId !== null) {
      return; // Prevent multiple loops
    }
    this.inferenceStatus = "Running inference...";
    this.requestUpdate();
    // Use a separate flag to control the loop, inferenceLoopId just stores the handle
    this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
  }

  private cancelInferenceLoop() {
    if (this.inferenceLoopId !== null) {
      cancelAnimationFrame(this.inferenceLoopId);
      this.inferenceLoopId = null;
      if (!this.modelError && !this.isModelLoading) {
        this.inferenceStatus = "Inference stopped.";
        // Clear canvas when stopping to remove lingering boxes/polygons
        this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
        this.requestUpdate();
      }
    }
  }

  private async runInference() {
     // --- Condition Checks ---
    if (this.isInferencing) {
        // If already processing, just schedule the next frame check
        this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
        return;
    }
    if (!this.tfjsModel || !this.currentStream || !this.videoElement || !this.ctx || this.videoElement.paused || this.videoElement.ended || this.videoElement.readyState < this.videoElement.HAVE_CURRENT_DATA) {
        // If conditions not met, schedule next check
        this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
        return;
    }

    // --- Start Processing ---
    this.isInferencing = true;

    // --- Clear Canvas (Fix Flickering) ---
    // Clear *before* any async operations or drawing for this frame
    // this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    // Update canvas size and scaling factors *before* this frame's drawing
    this.updateCanvasSize();

    const tensorsToDispose: tf.Tensor[] = [];
    let output: tf.Tensor[] | null = null;

    try {
        // 1. Preprocess Frame (tidy manages intermediate tensors)
        const inputTensor = tf.tidy(() => {
            const frame = tf.browser.fromPixels(this.videoElement);
            const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]);
            const normalized = resized.div(255.0);
            const batched = normalized.expandDims(0);
            return batched.cast('float32');
        });
        tensorsToDispose.push(inputTensor); // Track final input tensor

        // 2. Run Inference
        output = await this.tfjsModel!.executeAsync(inputTensor) as tf.Tensor[];
        if (output) {
          tensorsToDispose.push(...output);
        } // Track all output tensors

        // 3. Post-process & Draw
        if (output && output.length === 3) {
            const detections = output[0]; // Shape [1, num_detections, 38]
            // Note: output[1] and output[2] (masks and protos) are ignored as requested

            // Get detection data once
            const [detectionsBatch, _, __] = await detections.array() as number[][][];
            let detectionCount = 0;

            // --- Drawing Loop ---
          this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
            for (const detection of detectionsBatch) {
                const confidence = detection[4];
                if (confidence >= CONFIDENCE_THRESHOLD) {
                    detectionCount++;
                    const classId = Math.round(detection[5]);
                    const color = COLORS[classId % COLORS.length];

                    // Draw Bounding Box
                    this.drawBoundingBox(detection, color);
                }
            }
             this.inferenceStatus = `Detections: ${detectionCount}`;
        } else {
             console.warn("Model output was not the expected array of 3 tensors:", output); this.inferenceStatus = "Unexpected model output.";
        }

    } catch (error: any) {
        console.error("Error during inference:", error);
        this.inferenceStatus = `Inference Error: ${error.message || error}`;
    } finally {
        // --- Dispose ALL tracked tensors ---
        tf.dispose(tensorsToDispose);

        this.isInferencing = false; // Allow next inference
        this.requestUpdate(); // Update status display

        // --- Schedule Next Frame ---
        // Ensure loop continues even if errors occurred in processing
        if (this.inferenceLoopId !== null) { // Check if loop was cancelled by disconnect
             this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
        }
    }
  }

  // --- Helper to Draw Bounding Box ---
  private drawBoundingBox(detectionData: number[], color: number[]) {
    if (!this.ctx) {
      return;
    }

    const videoWidth = this.videoElement.videoWidth;
    const videoHeight = this.videoElement.videoHeight;
    const confidence = detectionData[4];
    const classId = Math.round(detectionData[5]);
    const x1 = detectionData[0], y1 = detectionData[1], x2 = detectionData[2], y2 = detectionData[3];

    // Scale box coordinates from model input size to video size
    const videoX1 = (x1 / MODEL_INPUT_WIDTH) * videoWidth;
    const videoY1 = (y1 / MODEL_INPUT_HEIGHT) * videoHeight;
    const videoX2 = (x2 / MODEL_INPUT_WIDTH) * videoWidth;
    const videoY2 = (y2 / MODEL_INPUT_HEIGHT) * videoHeight;
    // Scale box coordinates from video size to canvas size (considering 'contain' fit)
    const canvasX1 = videoX1 * this.scale + this.offsetX;
    const canvasY1 = videoY1 * this.scale + this.offsetY;
    const canvasWidth = (videoX2 - videoX1) * this.scale;
    const canvasHeight = (videoY2 - videoY1) * this.scale;

    // Draw Box
    this.ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(canvasX1, canvasY1, canvasWidth, canvasHeight);

    // Draw Label
    this.ctx.font = '12px sans-serif'; // Ensure font is set for accurate measurement
    this.ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.8)`; // Semi-transparent background
    const label = `Class ${classId}: ${confidence.toFixed(2)}`;
    const textMetrics = this.ctx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = 12; // Approximate height based on font size
    // Position background slightly above the box or inside if near top
    const textY = canvasY1 > textHeight + 4 ? canvasY1 - 2 : canvasY1 + textHeight + 2; // Adjusted y-pos logic
    this.ctx.fillRect(canvasX1 - 1, textY - textHeight - 1 , textWidth + 2, textHeight + 2); // Background rectangle
    this.ctx.fillStyle = `white`; // Text color
    this.ctx.fillText(label, canvasX1, textY - 1); // Adjusted y-pos logic for text baseline

  }

  // --- Render ---
  render() {
    let statusMessage = this.inferenceStatus;
    // More detailed status logic
    if (this.modelError) {
      statusMessage = `Error: ${this.modelError}`;
    } else if (this.isModelLoading) {
      statusMessage = "Loading model...";
    } else if (!this.currentStream) {
      statusMessage = "Waiting for camera access...";
    } else if (!this.tfjsModel) {
      statusMessage = "Model loaded, waiting for stream...";
    }

    return html`
      <video id="video" muted playsinline></video>
      <canvas id="overlay-canvas"></canvas>
      <div class="status">${statusMessage}</div>
    `;
  }
}
