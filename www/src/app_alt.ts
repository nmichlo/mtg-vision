// Single File: app.ts (or app.js if compiled)
import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, state, query, customElement } from 'lit/decorators.js';
import * as tf from '@tensorflow/tfjs';

// --- Configuration ---
const MODEL_URL = '/assets/models/yolov11s_seg__dk964hap__web_model/model.json'; // Ensure this path is correct for your server setup
const MODEL_INPUT_WIDTH = 640;
const MODEL_INPUT_HEIGHT = 640;
const CONFIDENCE_THRESHOLD = 0.5; // Filter detections below this score
const MASK_THRESHOLD = 0.5;       // Threshold for binary mask generation
const MASK_ALPHA = 0.4;           // Opacity for drawing masks
// --- Color Palette ---
const COLORS = [
  [255, 99, 71], [255, 165, 0], [255, 215, 0], [154, 205, 50],
  [60, 179, 113], [32, 178, 170], [70, 130, 180], [138, 43, 226],
  [218, 112, 214], [255, 105, 180]
];
// ---------------------


@customElement('video-container')
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
  @query('#video') private videoElement!: HTMLVideoElement;
  @query('#overlay-canvas') private canvasElement!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D;

  // --- Offscreen Canvas for Mask Rendering ---
  private maskCanvas: HTMLCanvasElement | null = null;
  private maskCtx: CanvasRenderingContext2D | null = null;

  // --- TFJS State ---
  @state() private tfjsModel: tf.GraphModel | null = null;
  @state() private isModelLoading: boolean = true;
  @state() private modelError: string | null = null;
  @state() private inferenceStatus: string = 'Initializing...';
  private protoMasksTensor: tf.Tensor | null = null; // Store preprocessed protos

  // --- Video Stream State ---
  private currentStream: MediaStream | null = null;
  private inferenceLoopId: number | null = null;
  private videoDisplayWidth: number = 0;
  private videoDisplayHeight: number = 0;
  private resizeObserver!: ResizeObserver;
  // No longer needed with synchronous predict: private isInferencing: boolean = false;


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
    tf.dispose(this.protoMasksTensor); // Dispose stored proto tensor
    this.tfjsModel?.dispose();
    this.tfjsModel = null;
    this.resizeObserver?.disconnect();
    console.log('TFJS model & protos disposed, observer disconnected.');
  }

  protected firstUpdated(_changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>): void {
    console.log('VideoContainer firstUpdated');
    if (!this.videoElement || !this.canvasElement) {
        this.inferenceStatus = "Initialization Error: Elements missing.";
        console.error(this.inferenceStatus);
        return;
    }

    this.ctx = this.canvasElement.getContext('2d')!;
    if (!this.ctx) {
      this.inferenceStatus = "Canvas context error.";
      console.error(this.inferenceStatus);
      return;
    }

    this.maskCanvas = document.createElement('canvas');
    this.maskCanvas.width = MODEL_INPUT_WIDTH; // Size for drawing upscaled mask
    this.maskCanvas.height = MODEL_INPUT_HEIGHT;
    this.maskCtx = this.maskCanvas.getContext('2d', { willReadFrequently: true });
    if (!this.maskCtx) {
        this.inferenceStatus = "Mask canvas context error.";
        console.error(this.inferenceStatus);
        return;
    }

    this.videoElement.addEventListener('loadedmetadata', this.handleVideoMetadataLoaded);
    this.videoElement.addEventListener('error', (e) => {
      console.error('Video element error:', e);
      this.inferenceStatus = 'Video element error.';
    });

    this.resizeObserver = new ResizeObserver(() => this.updateCanvasSize());
    this.resizeObserver.observe(this.videoElement);
    this.updateCanvasSize(); // Initial size
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
    this.isModelLoading = true;
    this.modelError = null;
    this.inferenceStatus = 'Loading model...';
    this.requestUpdate();
    console.log('Loading TFJS model from:', MODEL_URL);
    try {
      this.tfjsModel = await tf.loadGraphModel(MODEL_URL);

      // Warm up (now using predict)
      tf.tidy(() => {
           const dummyInput = tf.zeros([1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 3], 'float32');
           this.tfjsModel!.execute(dummyInput); // Synchronous predict/execute
      });
      console.log('TFJS model warmed up.');

      this.isModelLoading = false;
      this.inferenceStatus = 'Model loaded. Waiting for video...';
      console.log('TFJS model loaded successfully.');
      this.startInferenceLoop();
    } catch (error) {
      this.isModelLoading = false;
      this.modelError = `Failed to load model: ${error}`;
      this.inferenceStatus = this.modelError;
      console.error('TFJS model load failed:', error);
    } finally {
        this.requestUpdate();
    }
  }

  // --- Video Stream Methods ---
  private async startVideoStream() {
    console.log('Attempting to start video stream...');
    if (this.currentStream) this.stopVideoStream(); // Ensure clean state
    try {
      const constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 } } };
      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('getUserMedia successful');
      if (this.videoElement) {
        this.videoElement.srcObject = this.currentStream;
        this.inferenceStatus = this.isModelLoading ? 'Model loading...' : 'Video stream started.';
      } else {
           this.inferenceStatus = 'Video element not found.';
           console.warn(this.inferenceStatus);
      }
    } catch (error) {
      console.error('Failed to get user media:', error);
      this.inferenceStatus = `Camera access failed: ${error.message}`;
      this.currentStream = null;
      if (this.videoElement) this.videoElement.srcObject = null;
    } finally {
        this.requestUpdate();
    }
  }

  private stopVideoStream() {
    this.cancelInferenceLoop();
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop());
      console.log('MediaStream tracks stopped.');
    }
    this.currentStream = null;
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.srcObject = null;
      console.log('Video source cleared and paused.');
    }
    this.inferenceStatus = "Video stream stopped.";
    this.requestUpdate();
  }

  // --- Inference Loop ---
  private startInferenceLoop() {
    if (this.inferenceLoopId !== null) return; // Already running
    if (!this.tfjsModel || !this.currentStream) {
      console.log("Cannot start inference loop: Model or stream not ready.");
      return;
    }
    console.log("Starting inference loop.");
    this.inferenceStatus = "Running inference...";
    this.requestUpdate();
    this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
  }

  private cancelInferenceLoop() {
    if (this.inferenceLoopId !== null) {
      cancelAnimationFrame(this.inferenceLoopId);
      this.inferenceLoopId = null;
      console.log("Inference loop cancelled.");
       if (!this.modelError && !this.isModelLoading) {
           this.inferenceStatus = "Inference stopped.";
           this.requestUpdate();
       }
    }
  }

  private async runInference() {
    // --- Condition Checks ---
    // Removed isInferencing flag - predict is synchronous within RAF
    if (!this.tfjsModel || !this.currentStream || !this.videoElement || !this.ctx || !this.maskCtx || !this.maskCanvas || this.videoElement.paused || this.videoElement.ended || this.videoElement.readyState < this.videoElement.HAVE_CURRENT_DATA) {
      this.inferenceLoopId = requestAnimationFrame(() => this.runInference()); // Keep checking
      return;
    }

    this.updateCanvasSize(); // Ensure canvas size is current

    let output: tf.Tensor[] | null = null;
    let detectionData: number[][][] | null = null;
    const tempTensors: tf.Tensor[] = []; // For tensors needing disposal after async ops

    try {
      // --- Use tf.tidy for most tensor operations ---
      tf.tidy(() => {
        // 1. Capture and Preprocess Frame
        const frameTensor = tf.browser.fromPixels(this.videoElement);

        // Preprocessing within tidy
        const inputTensor = tf.tidy(() => {
            const resized = tf.image.resizeBilinear(frameTensor, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]);
            const normalized = resized.div(255.0);
            return normalized.expandDims(0).cast('float32');
        });
        // No need to push frameTensor, resized, normalized, batched to dispose list - tidy handles them

        // 2. Run Inference Synchronously
        // Use predict() - output is Tensor or Tensor[]
        const prediction = this.tfjsModel!.predict(inputTensor) as tf.Tensor[];

        // Store output tensors outside tidy to access their data later
        // Note: tf.tidy would dispose these if not returned.
        // Alternative: Keep execution inside tidy and return data directly (more complex).
        // We will dispose these manually in finally block.
        output = prediction;

        // Preprocess mask protos *once* if not already done
        if (!this.protoMasksTensor && output && output.length === 3) {
             const protoMasks = output[2]; // Shape [1, 160, 160, 32]
             const protoH = protoMasks.shape[1] || 160;
             const protoW = protoMasks.shape[2] || 160;
             const numProto = protoMasks.shape[3] || 32;
             // Reshape and keep this tensor outside tidy
             this.protoMasksTensor = tf.keep(protoMasks.squeeze(0).reshape([-1, numProto])); // Shape [protoH*protoW, numProto]
             console.log("Preprocessed and stored mask protos tensor.");
        }
      }); // --- End of main tf.tidy() ---


      // 3. Process Output Tensors (outside tidy, requires manual disposal of outputs)
      this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

      if (output && output.length === 3 && this.protoMasksTensor) {
        const detections = output[0]; // Shape [1, num_detections, 38]
        // Get data asynchronously
        detectionData = await detections.array() as number[][][];
        const detectionsBatch = detectionData[0];

        // Calculate scaling factors
        const videoWidth = this.videoElement.videoWidth;
        const videoHeight = this.videoElement.videoHeight;
        const scaleX = this.videoDisplayWidth / videoWidth;
        const scaleY = this.videoDisplayHeight / videoHeight;
        const scale = Math.min(scaleX, scaleY);
        const offsetX = (this.videoDisplayWidth - videoWidth * scale) / 2;
        const offsetY = (this.videoDisplayHeight - videoHeight * scale) / 2;

        let detectionCount = 0;
        const maskDrawingPromises: Promise<void>[] = [];

        for (const detection of detectionsBatch) {
          const confidence = detection[4];

          if (confidence >= CONFIDENCE_THRESHOLD) {
            detectionCount++;
            const classId = Math.round(detection[5]);
            const color = COLORS[classId % COLORS.length];
            const maskCoeffs = detection.slice(6);

            // Draw Bounding Box (sync)
            const x1 = detection[0], y1 = detection[1], x2 = detection[2], y2 = detection[3];
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

            // Generate and Draw Mask (async part)
            // Use a separate async function to handle mask generation and drawing
             maskDrawingPromises.push(
                this.generateAndDrawMask(
                    this.protoMasksTensor, // Use the stored tensor
                    maskCoeffs,
                    color, scale, offsetX, offsetY,
                    videoWidth, videoHeight
                )
            );
          } // end confidence check
        } // end detection loop

        // Wait for all mask drawing promises to complete
        await Promise.all(maskDrawingPromises);

        this.inferenceStatus = `Detections: ${detectionCount}`;

      } else { // Handle unexpected output or missing protos
        if (!this.protoMasksTensor && output && output.length === 3) {
            console.warn("Proto masks tensor not ready yet.");
            this.inferenceStatus = "Processing protos...";
        } else {
            console.warn("Model output not as expected or protos missing:", output);
            this.inferenceStatus = "Unexpected model output.";
        }
      }

    } catch (error) {
      console.error("Error during inference:", error);
      this.inferenceStatus = "Error during inference.";
    } finally {
      // Dispose the output tensors from predict() and any temp tensors
      tf.dispose(output); // Dispose the array returned by predict/executeAsync
      tf.dispose(tempTensors); // Dispose any tensors explicitly tracked
      // Note: protoMasksTensor is kept via tf.keep() and disposed in disconnectedCallback

      // Schedule Next Frame
      this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
      // Update UI at the end of the frame
      this.requestUpdate();
    }
  }

   // --- Helper to Generate and Draw Single Mask ---
  private async generateAndDrawMask(
        protoTensor: tf.Tensor, // Should be shape [protoH*protoW, numProto]
        coeffs: number[],
        color: number[],
        scale: number,
        offsetX: number,
        offsetY: number,
        videoWidth: number,
        videoHeight: number
      ): Promise<void> {

        if (!this.maskCtx || !this.maskCanvas || !this.ctx) return;

        let binaryMask: tf.Tensor | null = null; // To ensure disposal

        try {
            // Generate mask within a tidy scope
            binaryMask = tf.tidy(() => {
                const protoH = Math.sqrt(protoTensor.shape[0] || (160*160)); // Infer H from Area
                const protoW = protoH; // Assuming square protos
                const numProto = protoTensor.shape[1] || 32;

                const coeffsTensor = tf.tensor(coeffs, [numProto, 1]);
                // protosReshaped is passed in as protoTensor
                const maskProtoMul = tf.matMul(protoTensor, coeffsTensor);
                const maskSigmoid = tf.sigmoid(maskProtoMul).reshape([protoH, protoW]);

                // Upscale mask to model input size
                // FIX: Ensure input to resize is rank 3 or 4 by adding channel dim
                const maskSigmoid3D = maskSigmoid.expandDims(-1); // Add channel dim -> [protoH, protoW, 1]
                const upscaledMask = tf.image.resizeBilinear(maskSigmoid3D, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]);

                // Threshold and remove channel dim -> [MODEL_H, MODEL_W]
                return tf.greater(upscaledMask, MASK_THRESHOLD).squeeze([2]).cast('int32'); // Squeeze channel dim
            });

            // Get mask data (async)
            const maskData = await binaryMask.data() as Int32Array; // Expect 0 or 1

            // Create/clear ImageData for the offscreen canvas
            const maskWidth = binaryMask.shape[1] || MODEL_INPUT_WIDTH;
            const maskHeight = binaryMask.shape[0] || MODEL_INPUT_HEIGHT;
            // Check if offscreen canvas needs resize (shouldn't if model size is constant)
            if (this.maskCanvas.width !== maskWidth || this.maskCanvas.height !== maskHeight) {
                this.maskCanvas.width = maskWidth;
                this.maskCanvas.height = maskHeight;
            }
            const imageData = this.maskCtx.createImageData(maskWidth, maskHeight);
            const rgbaColor = [...color, MASK_ALPHA * 255];

            // Fill ImageData based on the mask
            for (let i = 0; i < maskData.length; i++) {
                if (maskData[i] === 1) { // If mask pixel is "on" (now 0 or 1)
                    const idx = i * 4;
                    imageData.data[idx] = rgbaColor[0];     // R
                    imageData.data[idx + 1] = rgbaColor[1]; // G
                    imageData.data[idx + 2] = rgbaColor[2]; // B
                    imageData.data[idx + 3] = rgbaColor[3]; // A
                }
            }
            // Put the mask onto the offscreen canvas
            this.maskCtx.putImageData(imageData, 0, 0);

            // Draw the offscreen canvas onto the main overlay canvas with scaling
            this.ctx.drawImage(
                this.maskCanvas,
                0, 0, maskWidth, maskHeight, // Source rect
                offsetX, offsetY,            // Destination pos
                videoWidth * scale, videoHeight * scale // Destination size
            );

        } catch(error) {
            console.error("Error generating/drawing mask:", error);
        } finally {
             tf.dispose(binaryMask); // Dispose the mask tensor created in tidy
        }
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

@customElement('app-container') // Define AppContainer custom element
class AppContainerElement extends LitElement { // Renamed to avoid conflict
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
