import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, state, query, customElement } from 'lit/decorators.js';
import * as tf from '@tensorflow/tfjs'; // Import TensorFlow.js

// ===========================================================
// App Container Component
// ===========================================================

class AppContainer extends LitElement {
  static styles = css`
    :host {
      display: flex; /* Use flex to allow video-container to grow */
      flex-direction: column; /* Stack elements vertically */
      width: 100%;
      height: 100%;
      overflow: hidden; /* Prevent scrollbars */
      font-family: sans-serif;
    }
    video-container {
      flex: 1; /* Allow video container to take available space */
      min-height: 0; /* Prevent flex item from overflowing */
    }
  `;

  render() {
    // Render only the video container
    return html`<video-container></video-container>`;
  }
}
customElements.define('app-container', AppContainer);

// ===========================================================
// Video Container Component
// ===========================================================

// --- Configuration ---
// !!! REPLACE WITH THE ACTUAL URL TO YOUR TFJS model.json !!!
const MODEL_URL = '/assets/models/yolov11s_seg__dk964hap__web_model/model.json';
// !!! ADJUST INPUT SIZE AND PREPROCESSING TO MATCH YOUR MODEL !!!
const MODEL_INPUT_WIDTH = 640;
const MODEL_INPUT_HEIGHT = 640;
const CONFIDENCE_THRESHOLD = 0.5; // Filter detections below this score
const MASK_THRESHOLD = 0.5;       // Threshold for binary mask generation
const MASK_ALPHA = 0.4;           // Opacity for drawing masks
// --- Color Palette (Example: You can generate more colors) ---
const COLORS = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    [255, 0, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0]
];
// ---------------------

@customElement('video-container') // Decorator to define the element
class VideoContainer extends LitElement {
  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      position: relative; /* For absolute positioning of canvas */
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
    /* Canvas styles */
    canvas#overlay-canvas { /* More specific selector */
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
  @query('#video') private videoElement!: HTMLVideoElement;
  @query('#overlay-canvas') private canvasElement!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D; // Assert non-null after firstUpdated

  // --- Offscreen Canvas for Mask Rendering ---
  private maskCanvas!: HTMLCanvasElement; // Initialize in firstUpdated
  private maskCtx!: CanvasRenderingContext2D; // Initialize in firstUpdated

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

  // --- Lit Lifecycle ---

  connectedCallback() {
    super.connectedCallback();
    console.log('VideoContainer connected');
    tf.ready().then(() => {
      console.log('TF.js backend ready:', tf.getBackend());
      this.loadTfjsModel();
      this.startVideoStream();
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    console.log('VideoContainer disconnected');
    this.stopVideoStream(); // Includes cancelling loop
    this.tfjsModel?.dispose();
    this.tfjsModel = null;
    this.resizeObserver?.disconnect();
    console.log('TFJS model disposed, observer disconnected.');
  }

  protected firstUpdated(_changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>): void {
    console.log('VideoContainer firstUpdated');
    if (!this.videoElement || !this.canvasElement) {
        console.error("Required elements not found!");
        this.inferenceStatus = "Initialization Error: Elements missing.";
        this.requestUpdate(); return;
    }

    const context = this.canvasElement.getContext('2d');
    if (!context) {
      console.error("Failed to get 2D context for overlay canvas");
      this.inferenceStatus = "Canvas context error.";
       this.requestUpdate(); return;
    }
    this.ctx = context;

    // Initialize offscreen canvas
    this.maskCanvas = document.createElement('canvas');
    this.maskCanvas.width = MODEL_INPUT_WIDTH;
    this.maskCanvas.height = MODEL_INPUT_HEIGHT;
    const maskContext = this.maskCanvas.getContext('2d', { willReadFrequently: true });
     if (!maskContext) {
        console.error("Failed to get 2D context for mask canvas");
        this.inferenceStatus = "Mask canvas context error.";
        this.requestUpdate(); return;
    }
    this.maskCtx = maskContext;

    this.videoElement.addEventListener('loadedmetadata', this.handleVideoMetadataLoaded);
    this.videoElement.addEventListener('error', (e) => {
      console.error('Video element error:', e); this.inferenceStatus = 'Video element error.'; this.requestUpdate();
    });

    this.resizeObserver = new ResizeObserver(() => this.updateCanvasSize());
    this.resizeObserver.observe(this.videoElement);
    this.updateCanvasSize();
  }

  // --- Event Handlers ---
  private handleVideoMetadataLoaded = () => {
    console.log('Video metadata loaded');
    this.videoElement.play().catch(e => console.error('Video play failed:', e));
    this.updateCanvasSize();
  }

  // --- Size & Scaling ---
  private updateCanvasSize() {
    if (!this.videoElement || !this.canvasElement) return;
    const displayWidth = this.videoElement.clientWidth;
    const displayHeight = this.videoElement.clientHeight;
    if (this.canvasElement.width !== displayWidth || this.canvasElement.height !== displayHeight) {
      this.canvasElement.width = displayWidth;
      this.canvasElement.height = displayHeight;
      this.videoDisplayWidth = displayWidth;
      this.videoDisplayHeight = displayHeight;
      console.log(`Canvas resized to: ${displayWidth}x${displayHeight}`);
    }
  }

  // --- TFJS Methods ---
  private async loadTfjsModel() {
    this.isModelLoading = true; this.modelError = null; this.inferenceStatus = 'Loading model...'; this.requestUpdate();
    console.log('Loading TFJS model from:', MODEL_URL);
    try {
      this.tfjsModel = await tf.loadGraphModel(MODEL_URL);
      // Warm up
      const dummyInput = tf.zeros([1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 3], 'float32');
      const warmupResult = await this.tfjsModel.executeAsync(dummyInput) as tf.Tensor[];
      tf.dispose(dummyInput); tf.dispose(warmupResult);
      console.log('TFJS model warmed up.');

      this.isModelLoading = false; this.inferenceStatus = 'Model loaded. Waiting for video...'; console.log('TFJS model loaded successfully.');
      this.startInferenceLoop();
    } catch (error) {
      this.isModelLoading = false; this.modelError = `Failed to load model: ${error}`; this.inferenceStatus = this.modelError; console.error('TFJS model load failed:', error);
    } finally { this.requestUpdate(); }
  }

  // --- Video Stream Methods ---
  private async startVideoStream() {
    console.log('Attempting to start video stream...'); this.inferenceStatus = 'Requesting camera...'; this.requestUpdate();
    if (this.currentStream) { console.log('Stopping existing stream first.'); this.stopVideoStream(); }
    try {
      const constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 } } };
      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('getUserMedia successful');
      if (this.videoElement) {
        this.videoElement.srcObject = this.currentStream;
        this.inferenceStatus = this.isModelLoading ? 'Model loading...' : 'Video stream started.';
      } else { console.warn('Video element not ready when stream started.'); this.inferenceStatus = 'Video element not found.'; }
    } catch (error) {
      console.error('Failed to get user media:', error); this.inferenceStatus = `Camera access failed: ${error.message}`; this.currentStream = null;
      if (this.videoElement) this.videoElement.srcObject = null;
    } finally { this.requestUpdate(); }
  }

  private stopVideoStream() {
    this.cancelInferenceLoop();
    if (this.currentStream) { this.currentStream.getTracks().forEach(track => track.stop()); console.log('MediaStream tracks stopped.'); }
    this.currentStream = null;
    if (this.videoElement) { this.videoElement.pause(); this.videoElement.srcObject = null; console.log('Video source cleared and paused.'); }
    this.inferenceStatus = "Video stream stopped."; this.requestUpdate();
  }

  // --- Inference Loop ---
  private startInferenceLoop() {
    if (this.inferenceLoopId !== null) return;
    // if (!this.tfjsModel || !this.currentStream) { console.log("Cannot start inference loop: Model or stream not ready."); return; }
    console.log("Starting inference loop."); this.inferenceStatus = "Running inference..."; this.requestUpdate();
    this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
  }

  private cancelInferenceLoop() {
    if (this.inferenceLoopId !== null) {
      cancelAnimationFrame(this.inferenceLoopId); this.inferenceLoopId = null; console.log("Inference loop cancelled.");
      if (!this.modelError && !this.isModelLoading) { this.inferenceStatus = "Inference stopped."; this.requestUpdate(); }
    }
  }

  private async runInference() {
    // --- Condition Checks ---
    if (this.isInferencing) {
      this.inferenceLoopId = requestAnimationFrame(() => this.runInference()); return;
    }
    if (!this.tfjsModel || !this.currentStream || !this.videoElement || !this.ctx || !this.maskCtx || !this.maskCanvas || this.videoElement.paused || this.videoElement.ended || this.videoElement.readyState < this.videoElement.HAVE_CURRENT_DATA) {
      this.inferenceLoopId = requestAnimationFrame(() => this.runInference()); return;
    }

    this.isInferencing = true;
    this.updateCanvasSize();

    const tensorsToDispose: tf.Tensor[] = [];
    let output: tf.Tensor[] | null = null;

    try {
      // 1. Capture and Preprocess Frame within tf.tidy
      const inputTensor = tf.tidy(() => {
        const frame = tf.browser.fromPixels(this.videoElement);
        tensorsToDispose.push(frame); // Add original frame for potential later use if needed, dispose outside tidy
        const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]);
        const normalized = resized.div(255.0);
        const batched = normalized.expandDims(0);
        return batched.cast('float32'); // Return final input
      });
      tensorsToDispose.push(inputTensor); // Track input tensor for disposal

      // 2. Run Inference Asynchronously
      // console.time("inference"); // Optional: time inference
      output = await this.tfjsModel!.executeAsync(inputTensor) as tf.Tensor[];
      // console.timeEnd("inference");
      if (output) tensorsToDispose.push(...output); // Track output tensors

      // 3. Process Output Tensors
      this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

      if (output && output.length === 3) {
        const detections = output[0]; // Shape [1, num_detections, 38]
        // const protoMasks = output[2]; // Shape [1, 160, 160, 32]

        const detectionData = await detections.array() as number[][][];
        const detectionsBatch = detectionData[0];

        // const protoH = protoMasks.shape[1] || 160;
        // const protoW = protoMasks.shape[2] || 160;
        // const numProto = protoMasks.shape[3] || 32;

        // Reshape protos *once* outside the loop for efficiency
        // const protosReshaped = tf.tidy(() => protoMasks.squeeze(0).reshape([-1, numProto])); // [protoH*protoW, numProto]
        // tensorsToDispose.push(protosReshaped); // Track for final disposal

        // Calculate scaling factors
        const videoWidth = this.videoElement.videoWidth;
        const videoHeight = this.videoElement.videoHeight;
        const scaleX = this.videoDisplayWidth / videoWidth;
        const scaleY = this.videoDisplayHeight / videoHeight;
        const scale = Math.min(scaleX, scaleY);
        const offsetX = (this.videoDisplayWidth - videoWidth * scale) / 2;
        const offsetY = (this.videoDisplayHeight - videoHeight * scale) / 2;

        let detectionCount = 0;
        const drawingPromises: Promise<void>[] = [];

        for (const detection of detectionsBatch) {
          const confidence = detection[4];
          if (confidence >= CONFIDENCE_THRESHOLD) {
            detectionCount++;
            const classId = Math.round(detection[5]);
            const color = COLORS[classId % COLORS.length];
            const maskCoeffs = detection.slice(6); // Coeffs for this detection

            // --- Generate Mask Tensor within a tf.tidy scope ---
            // const binaryMaskTensor = tf.tidy(() => {
            //   const coeffsTensor = tf.tensor(maskCoeffs, [numProto, 1]); // [32, 1]
            //   const maskProtoMul = tf.matMul(protosReshaped, coeffsTensor); // [protoH*protoW, 1]
            //   const maskSigmoid = tf.sigmoid(maskProtoMul).reshape([protoH, protoW]); // [protoH, protoW]
            //
            //   // Expand dims before resize: [protoH, protoW, 1]
            //   const mask3D = maskSigmoid.expandDims(-1);
            //   // Upscale mask to model input size
            //   const upscaledMask = tf.image.resizeBilinear(mask3D, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]);
            //
            //   // Threshold and remove channel dim: [MODEL_H, MODEL_W]
            //   return tf.greater(upscaledMask, MASK_THRESHOLD).squeeze([2]);
            // });
            // binaryMaskTensor will be automatically disposed by the outer finally{} block

            // --- Draw Mask & Box (asynchronously) ---
            // Pass the tensor to drawMask; it will handle .data() and disposal
            drawingPromises.push(this.drawDetection(
                detection, color, scale, offsetX, offsetY, videoWidth, videoHeight
            ));

             // IMPORTANT: Add the generated mask tensor to the list for disposal
            // tensorsToDispose.push(binaryMaskTensor);

          } // end confidence check
        } // end detection loop

        await Promise.all(drawingPromises); // Wait for all drawing to complete
        this.inferenceStatus = `Detections: ${detectionCount}`;

      } else { // Handle unexpected output format
        console.warn("Model output was not the expected array of 3 tensors:", output); this.inferenceStatus = "Unexpected model output.";
      }

    } catch (error: any) { // Catch specific errors if needed
      console.error("Error during inference:", error); this.inferenceStatus = `Inference Error: ${error.message || error}`;
    } finally {
      // --- Dispose ALL tracked tensors ---
      tf.dispose(tensorsToDispose);
      // No need to dispose 'output' separately if its contents were pushed to tensorsToDispose

      this.isInferencing = false; // Allow next inference
      this.requestUpdate(); // Update status display

      // --- Schedule Next Frame ---
      this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
    }
  }

   // --- Helper to Draw Single Detection (Mask + Box) ---
  private async drawDetection(
        detectionData: number[], // Single detection row [box, conf, class, coeffs...]
        // binaryMaskTensor: tf.Tensor, // The generated binary mask [H, W]
        color: number[],
        scale: number, offsetX: number, offsetY: number,
        videoWidth: number, videoHeight: number
      ): Promise<void> {

    if (!this.maskCtx || !this.maskCanvas || !this.ctx) return;

    // const maskWidth = binaryMaskTensor.shape[1] || MODEL_INPUT_WIDTH;
    // const maskHeight = binaryMaskTensor.shape[0] || MODEL_INPUT_HEIGHT;

    // Get mask data as a flat array (Uint8Array)
    // const maskData = await binaryMaskTensor.data(); // No need for specific type here

    // Create ImageData for the offscreen canvas
    // const imageData = this.maskCtx.createImageData(maskWidth, maskHeight);
    const rgbaColor = [...color, MASK_ALPHA * 255]; // Add alpha

    // Fill ImageData based on the mask
    // for (let i = 0; i < maskData.length; i++) {
    //     if (maskData[i] === 1) { // If mask pixel is "on"
    //         const idx = i * 4;
    //         imageData.data[idx]     = rgbaColor[0]; // R
    //         imageData.data[idx + 1] = rgbaColor[1]; // G
    //         imageData.data[idx + 2] = rgbaColor[2]; // B
    //         imageData.data[idx + 3] = rgbaColor[3]; // A
    //     }
    // }

    // Put the mask onto the offscreen canvas
    // this.maskCtx.putImageData(imageData, 0, 0);

    // Draw the offscreen canvas (mask) onto the main overlay canvas, scaled correctly
    // this.ctx.drawImage(
    //     this.maskCanvas,
    //     0, 0, maskWidth, maskHeight, // Source rect (full offscreen canvas)
    //     offsetX, offsetY,            // Destination top-left (with offset)
    //     videoWidth * scale, videoHeight * scale // Destination size (scaled video size)
    // );

    // --- Draw Bounding Box (on top of mask) ---
    const confidence = detectionData[4];
    const classId = Math.round(detectionData[5]);
    const x1 = detectionData[0], y1 = detectionData[1], x2 = detectionData[2], y2 = detectionData[3];
    const videoX1 = (x1 / MODEL_INPUT_WIDTH) * videoWidth;
    const videoY1 = (y1 / MODEL_INPUT_HEIGHT) * videoHeight;
    const videoX2 = (x2 / MODEL_INPUT_WIDTH) * videoWidth;
    const videoY2 = (y2 / MODEL_INPUT_HEIGHT) * videoHeight;
    const canvasX1 = videoX1 * scale + offsetX;
    const canvasY1 = videoY1 * scale + offsetY;
    const canvasWidth = (videoX2 - videoX1) * scale;
    const canvasHeight = (videoY2 - videoY1) * scale;

    this.ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(canvasX1, canvasY1, canvasWidth, canvasHeight);
    this.ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    this.ctx.fillText(
        `Class ${classId}: ${confidence.toFixed(2)}`,
        canvasX1, canvasY1 > 10 ? canvasY1 - 5 : 10
    );

    // --- Draw coefficients (optional) ---
    this.ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    this.ctx.beginPath();
    for (let i = 6; i < detectionData.length; i += 2) {
      const coeffx = detectionData[i*2];
      const coeffy = detectionData[i*2+1];
      const x0 = Math.min(x1, x2)
      const y0 = Math.min(y1, y2)
      const w = Math.abs(x2 - x1);
      const h = Math.abs(y2 - y1);
      const x = (x0 + coeffx * w) * scale + offsetX;
      const y = (y0 + coeffy * h) * scale + offsetY;
      this.ctx.lineTo(x, y);
    }
    this.ctx.closePath();
    this.ctx.stroke();
  }


  // --- Render ---

  render() {
    let statusMessage = this.inferenceStatus;
    if (this.modelError) statusMessage = `Error: ${this.modelError}`;
    else if (this.isModelLoading) statusMessage = "Loading model...";
    else if (!this.currentStream) statusMessage = "Waiting for camera access...";
    else if (!this.tfjsModel) statusMessage = "Model loaded, waiting for stream...";

    return html`
      <video id="video" muted playsinline></video>
      <canvas id="overlay-canvas"></canvas>
      <div class="status">${statusMessage}</div>
    `;
  }
}
