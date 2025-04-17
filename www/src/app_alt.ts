import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, state, query, customElement } from 'lit/decorators.js';
import * as tf from '@tensorflow/tfjs';

// Import Norfair types and classes
import { Tracker, TrackedObject, TrackerOptions, Point } from "./norfair"; // Added Point

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
  private tracker: Tracker | null = null; // Norfair tracker instance

  // --- Lit Lifecycle ---

  connectedCallback() {
    super.connectedCallback();
    tf.ready().then(() => {
      this.loadTfjsModel();
      this.startVideoStream();
      // Initialize the Norfair tracker
      this.tracker = new Tracker({
          distanceThreshold: 100, // Adjust based on object speed and frame rate
          hitInertiaMin: 0,      // Lowered defaults slightly, adjust as needed
          hitInertiaMax: 15,  //
          initDelay: 2,  // Number of frames to wait before starting tracking
      });
      console.log("Norfair tracker initialized.");
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.stopVideoStream();
    this.tfjsModel?.dispose();
    this.tfjsModel = null;
    this.resizeObserver?.disconnect();
    this.tracker = null; // Clear tracker state
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

    // Use ResizeObserver on the *video element* as the reference for sizing
    this.resizeObserver = new ResizeObserver(() => this.updateCanvasSize());
    this.resizeObserver.observe(this.videoElement);

    // Initial size update
    this.updateCanvasSize();
  }

  // --- Event Handlers ---
  private handleVideoMetadataLoaded = () => {
    // Ensure video dimensions are available before playing and sizing
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
    if (!this.videoElement || !this.canvasElement || !this.videoElement.videoWidth) {
      return; // Need video dimensions
    }

    const displayWidth = this.videoElement.clientWidth;
    const displayHeight = this.videoElement.clientHeight;

    // Set canvas drawing buffer size to match display size
    if (this.canvasElement.width !== displayWidth || this.canvasElement.height !== displayHeight) {
        this.canvasElement.width = displayWidth;
        this.canvasElement.height = displayHeight;
        this.videoDisplayWidth = displayWidth;
        this.videoDisplayHeight = displayHeight;
        console.log(`Canvas resized to: ${displayWidth}x${displayHeight}`);
    }

    // Calculate scaling factors based on video's intrinsic size and display size ('contain' logic)
    const videoWidth = this.videoElement.videoWidth;
    const videoHeight = this.videoElement.videoHeight;
    const scaleX = this.videoDisplayWidth / videoWidth;
    const scaleY = this.videoDisplayHeight / videoHeight;
    this.scale = Math.min(scaleX, scaleY); // Use minimum scale for 'contain'

    // Calculate offsets to center the scaled video drawing on the canvas
    this.offsetX = (this.videoDisplayWidth - videoWidth * this.scale) / 2;
    this.offsetY = (this.videoDisplayHeight - videoHeight * this.scale) / 2;
    // console.log(`Video: ${videoWidth}x${videoHeight}, Scale: ${this.scale.toFixed(3)}, Offset: ${this.offsetX.toFixed(1)}, ${this.offsetY.toFixed(1)}`);
  }


  // --- TFJS Methods ---
  private async loadTfjsModel() {
    this.isModelLoading = true; this.modelError = null; this.inferenceStatus = 'Loading model...'; this.requestUpdate();
    try {
      this.tfjsModel = await tf.loadGraphModel(MODEL_URL);
      this.isModelLoading = false;
      this.inferenceStatus = 'Model loaded. Waiting for video...';
      console.log('TFJS model loaded successfully.');
      this.startInferenceLoop(); // Start inference only after model is loaded
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
      this.stopVideoStream(); // Stop existing stream first
    }
    try {
      // Request a resolution closer to the model input if possible
      const constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 } } };
      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      if (this.videoElement) {
        this.videoElement.srcObject = this.currentStream;
        // Don't start inference here, wait for model load and metadata
        this.inferenceStatus = this.isModelLoading ? 'Model loading...' : 'Video stream started. Waiting for model/metadata...';
      } else {
        this.inferenceStatus = 'Video element not ready.';
      }
    } catch (error: any) {
      this.inferenceStatus = `Camera failed: ${error.message}`;
      console.error('getUserMedia failed:', error);
      this.currentStream = null;
      if (this.videoElement) {
        this.videoElement.srcObject = null;
      }
    } finally {
      this.requestUpdate();
    }
  }

  private stopVideoStream() {
    this.cancelInferenceLoop(); // Stop inference first
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop());
    }
    this.currentStream = null;
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.srcObject = null;
    }
    // Clear canvas when stream stops
    this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
    this.inferenceStatus = "Video stream stopped.";
    this.requestUpdate();
  }

  // --- Inference Loop ---
  private startInferenceLoop() {
    // make sure not already running
    // It is OK if we start the loop and the model is not ready yet, the loop
    // will just skip frames in that case until it is ready.
    if (this.inferenceLoopId !== null || this.isInferencing) {
      console.log("Inference loop start condition not met.");
      return;
    }
    this.inferenceStatus = "Running inference...";
    this.isInferencing = false; // Reset flag
    console.log("Starting inference loop...");
    this.requestUpdate();
    this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
  }

  private cancelInferenceLoop() {
    if (this.inferenceLoopId !== null) {
      cancelAnimationFrame(this.inferenceLoopId);
      this.inferenceLoopId = null;
      this.isInferencing = false; // Ensure flag is reset
      if (!this.modelError && !this.isModelLoading) {
        this.inferenceStatus = "Inference stopped.";
        // Clear canvas when stopping to remove lingering boxes/polygons
        this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
        this.requestUpdate();
      }
      console.log("Inference loop cancelled.");
    }
  }

  private async runInference() {
     // --- Condition Checks ---
    if (this.isInferencing) {
        console.log("Already inferencing, skipping frame.");
        this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
        return;
    }
    if (!this.tfjsModel || !this.currentStream || !this.videoElement || !this.ctx || this.videoElement.paused || this.videoElement.ended || this.videoElement.readyState < this.videoElement.HAVE_CURRENT_DATA) {
        console.log("Inference conditions not met, scheduling next check.");
        this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
        return;
    }

    // --- Start Processing ---
    this.isInferencing = true;
    // console.log("Running inference frame.");

    // Update canvas size and scaling factors *before* this frame's drawing
    // This ensures drawing coordinates are based on the current layout
    this.updateCanvasSize();

    const tensorsToDispose: tf.Tensor[] = [];
    let output: tf.Tensor[] | null = null;

    try {
        // 1. Preprocess Frame (tidy manages intermediate tensors)
        const inputTensor = tf.tidy(() => {
            const frame = tf.browser.fromPixels(this.videoElement);
            // Ensure resizing happens correctly
            const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]);
            const normalized = resized.div(255.0);
            const batched = normalized.expandDims(0);
            return batched.cast('float32');
        });
        tensorsToDispose.push(inputTensor); // Track final input tensor for disposal

        // 2. Run Inference
        output = await this.tfjsModel!.executeAsync(inputTensor) as tf.Tensor[];
        if (output) {
          tensorsToDispose.push(...output); // Track all output tensors for disposal
        }

        // 3. Post-process, Track & Draw
        if (output && output.length === 3) { // Check for YOLOv11 output structure
            const detectionsTensor = output[0]; // Shape [1, num_detections, 38] (boxes, conf, class, masks...)

            // Get detection data once (await the promise)
            const detectionsBatch = (await detectionsTensor.array() as number[][][])[0]; // Get the first (only) batch

            // --- Prepare detections for Norfair ---
            const norfairDetections: Point[] = []; // Array to hold centroids [x, y]
            const originalDetectionData: { [index: number]: number[] } = {}; // Map Norfair index to original full detection data

            for (let i = 0; i < detectionsBatch.length; i++) {
                const detection = detectionsBatch[i];
                // Indices based on typical YOLOv8/YOLOv11 output:
                // 0-3: bbox [x1, y1, x2, y2] (in model input resolution)
                // 4: confidence score
                // 5: class id
                // 6+: mask info (ignored here)
                const confidence = detection[4];

                if (confidence >= CONFIDENCE_THRESHOLD) {
                    const x1 = detection[0], y1 = detection[1], x2 = detection[2], y2 = detection[3];
                    // Calculate centroid (center point of the bounding box)
                    const centerX = (x1 + x2) / 2;
                    const centerY = (y1 + y2) / 2;
                    const point: Point = [centerX, centerY]; // Norfair expects [x, y]

                    // Store the centroid for Norfair and map its index back to the original data
                    const norfairIndex = norfairDetections.length;
                    norfairDetections.push(point);
                    originalDetectionData[norfairIndex] = detection;
                }
            }
            // --------------------------------------

            // --- Update Norfair Tracker ---
            let trackingResults: number[] = [];
            if (this.tracker && norfairDetections.length > 0) {
                console.log("Updating tracker with detections:", norfairDetections);
                trackingResults = this.tracker.update(norfairDetections); // Update returns assigned IDs
                console.log("Tracker results:", trackingResults);
            } else if (this.tracker) {
                // Update tracker even with no detections to decrement hit counters
                trackingResults = this.tracker.update([]);
            }
            // ------------------------------

            // --- Drawing Loop ---
            // Clear canvas *before* drawing new frame elements
            this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
            let trackedDetectionCount = 0;

            // Draw boxes for currently detected & tracked objects
            for (let i = 0; i < norfairDetections.length; i++) {
                const assignedId = trackingResults[i]; // Get the ID assigned by Norfair
                const originalDetData = originalDetectionData[i]; // Get the full original detection data

                if (originalDetData) { // Should always be true if logic is correct
                    trackedDetectionCount++;
                    // Use ID for color, fallback for unassigned (-1)
                    const colorIndex = assignedId >= 0 ? assignedId % COLORS.length : COLORS.length - 1; // Use last color for initializing
                    const color = COLORS[colorIndex];

                    // Draw the bounding box using original data and include the assigned ID
                    this.drawBoundingBoxWithId(originalDetData, color, assignedId);
                }
            }

            // Optionally, draw estimates for objects tracked but not detected in this frame
            // this.drawUnmatchedEstimates(); // Implement this if needed see previous response

            // Update status message
            this.inferenceStatus = `Tracked Detections: ${trackedDetectionCount}`;

        } else {
             console.warn("Model output was not the expected array of 3 tensors:", output);
             this.inferenceStatus = "Unexpected model output.";
             // Clear canvas if output is bad
             this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        }

    } catch (error: any) {
        console.error("Error during inference:", error);
        this.inferenceStatus = `Inference Error: ${error.message || error}`;
        // Optionally clear canvas on error
        this.ctx?.clearRect(0, 0, this.canvasElement?.width ?? 0, this.canvasElement?.height ?? 0);
    } finally {
        // --- Dispose ALL tracked tensors ---
        tf.dispose(tensorsToDispose);

        this.isInferencing = false; // Allow next inference run
        this.requestUpdate(); // Update status display if needed

        // --- Schedule Next Frame ---
        // Ensure loop continues even if errors occurred in processing this frame
        if (this.inferenceLoopId !== null) { // Check if loop was cancelled (e.g., by disconnect)
             this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
        }
    }
  }

  // --- Helper to Draw Bounding Box WITH ID ---
  private drawBoundingBoxWithId(detectionData: number[], color: number[], trackId: number) {
    if (!this.ctx || !this.videoElement.videoWidth) { // Ensure video dimensions are available
      return;
    }

    const videoWidth = this.videoElement.videoWidth;
    const videoHeight = this.videoElement.videoHeight;
    const confidence = detectionData[4];
    const classId = Math.round(detectionData[5]);
    // Bbox coordinates are relative to the MODEL_INPUT size
    const x1 = detectionData[0], y1 = detectionData[1], x2 = detectionData[2], y2 = detectionData[3];

    // --- Coordinate Scaling ---
    // 1. Scale box coordinates from model input size (e.g., 640x640) to original video size
    const videoX1 = (x1 / MODEL_INPUT_WIDTH) * videoWidth;
    const videoY1 = (y1 / MODEL_INPUT_HEIGHT) * videoHeight;
    const videoX2 = (x2 / MODEL_INPUT_WIDTH) * videoWidth;
    const videoY2 = (y2 / MODEL_INPUT_HEIGHT) * videoHeight;

    // 2. Scale box coordinates from video size to canvas size (considering 'contain' fit and offsets)
    const canvasX1 = videoX1 * this.scale + this.offsetX;
    const canvasY1 = videoY1 * this.scale + this.offsetY;
    const canvasWidth = (videoX2 - videoX1) * this.scale;
    const canvasHeight = (videoY2 - videoY1) * this.scale;
    // --- End Scaling ---

    // Draw Box
    this.ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(canvasX1, canvasY1, canvasWidth, canvasHeight);

    // Draw Label with ID
    this.ctx.font = '12px sans-serif'; // Ensure font is set each time
    this.ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.8)`; // Background color

    // Create label text
    const idLabel = trackId === -1 ? 'Init' : `ID: ${trackId}`; // Show 'Init' for initializing objects
    const label = `${idLabel} (${confidence.toFixed(2)})`; // Removed Class ID for brevity, add back if needed

    // Calculate text size and position
    const textMetrics = this.ctx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = 12; // Approximate height based on font size
    const padding = 2;

    // Position background slightly above the box or inside if near top edge
    let textY = canvasY1 - padding - 1; // Default position above box
    let backgroundY = textY - textHeight - padding;

    if (backgroundY < 0) { // If label goes off the top, position it inside the box
        textY = canvasY1 + textHeight + padding;
        backgroundY = canvasY1 + padding;
    }

    // Draw background rectangle
    this.ctx.fillRect(canvasX1 - 1, backgroundY , textWidth + (padding * 2), textHeight + (padding * 2));

    // Draw text
    this.ctx.fillStyle = `white`; // Text color
    this.ctx.fillText(label, canvasX1 + padding -1 , textY); // Draw text inside background padding

  }

  // --- Render ---
  render() {
    // Determine the status message based on the current state
    let statusMessage = this.inferenceStatus;
    if (this.modelError) {
      statusMessage = `Error: ${this.modelError}`;
    } else if (this.isModelLoading) {
      statusMessage = "Loading model...";
    } else if (!this.currentStream) {
      statusMessage = "Waiting for camera access...";
    } else if (!this.tfjsModel) { // Model loaded but stream might not be ready
      statusMessage = "Model loaded, waiting for stream...";
    } else if (this.videoElement && this.videoElement.readyState < this.videoElement.HAVE_METADATA) {
        statusMessage = "Waiting for video metadata...";
    } else if (this.inferenceLoopId === null) { // Stream/model ready, but loop not running (e.g., initial state)
        statusMessage = "Ready to start inference.";
    }
    // If inference is running, this.inferenceStatus (e.g., "Tracked Detections: X") will be shown

    return html`
      <video id="video" muted playsinline></video>
      <canvas id="overlay-canvas"></canvas>
      <div class="status">${statusMessage}</div>
    `;
  }
}
