// ===========================================================
// Single File Refactored Application
// ===========================================================

import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, state, query, customElement } from 'lit/decorators.js';
import * as tf from '@tensorflow/tfjs';

// --- Norfair Imports ---
// Assuming Norfair classes (Tracker, TrackedObject, Point) are correctly imported or available globally
// If using a direct script include, you might need: declare var Tracker: any; etc.
import {Tracker, TrackedObject, TrackerOptions, Point} from "./norfair"; // Adjust path if needed

// ===========================================================
// Type Definitions
// ===========================================================

interface BBox { x1: number; y1: number; x2: number; y2: number; } // Usually in video coordinates
interface BBoxCanvas { x: number; y: number; w: number; h: number; } // Canvas coordinates
interface VideoDims { w: number; h: number; }

// Raw YOLO detection output structure
interface RawDetection {
  bboxModel: number[]; // [x1, y1, x2, y2] in model input coords (e.g., 0-640)
  confidence: number;
  classId: number;
  maskCoeffs: tf.Tensor; // Shape [1, 1, num_coeffs], owned by caller after return
}

// Information needed by the renderer for each object
interface DrawInfo {
  id: number;          // Track ID (-1 for initialization)
  color: number[];     // [r, g, b]
  bboxCanvas: BBoxCanvas; // Bounding box in final canvas coordinates
  label: string;       // Text label (e.g., "ID: 5 (0.88)")
  mask?: {             // Optional mask info
    tensor: tf.Tensor2D; // The generated binary mask tensor
    videoBox: BBox;      // The video coordinate box this mask corresponds to (for positioning/scaling)
  }
}

// Stored info per tracked object
interface ObjectEmbeddingInfo {
    id: number;
    hasBeenEmbedded: boolean;
    lastEmbeddingTime: number | null;
    creationTime: number;
    embedding: tf.Tensor | null; // Store the embedding tensor
    lastKnownBboxVideo: BBox | null; // Last known bounding box in *video coordinates*
}

// ===========================================================
// Constants
// ===========================================================

// --- YOLO Configuration ---
const MODEL_URL = '/assets/models/yolov11s_seg__dk964hap__web_model/model.json';
const MODEL_INPUT_WIDTH = 640;
const MODEL_INPUT_HEIGHT = 640;
const CONFIDENCE_THRESHOLD = 0.5;
const MASK_THRESHOLD = 0.5;
const PROTO_MASK_SIZE = 160;
const MASK_COEFF_COUNT = 32; // Should match model output (e.g., last 32 elements of detection)
const DET_BOX_COEFF_INDEX = 6; // Index where mask coefficients start in the detection output vector

// --- EMBED Configuration ---
const EMBED_URL = '/assets/models/convnextv2_convlinear__aivb8jvk-47500__encoder__web_model/model.json';
const EMBED_INPUT_WIDTH = 128;
const EMBED_INPUT_HEIGHT = 192;
const EMBEDDING_LOOP_INTERVAL_MS = 250;
const EMBEDDING_CROP_PADDING_FACTOR = 0.1;

// --- Drawing Configuration ---
const MASK_DRAW_ALPHA = 0.5;
// Keep padding at 0.0 to test core scaling first. Re-introduce (e.g., 0.02) if needed later.
const MASK_CROP_PADDING_FACTOR = 0.0;

// --- Color Palette ---
const COLORS = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    [255, 0, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
    [128, 0, 128], [0, 128, 128], [255, 165, 0], [255, 192, 203], [75, 0, 130]
];

// ===========================================================
// Service Class Implementations
// ===========================================================

// --- 1. VideoSourceManager ---
class VideoSourceManager {
    private videoElement: HTMLVideoElement | null = null;
    private currentStream: MediaStream | null = null;
    private streamError: string | null = null;

    constructor() {}

    async startStream(element: HTMLVideoElement, constraints?: MediaStreamConstraints): Promise<void> {
        this.videoElement = element;
        this.streamError = null;
        if (this.currentStream) {
            this.stopStream();
        }
        try {
            const defaultConstraints = { video: { width: { ideal: 640 }, height: { ideal: 480 } } };
            this.currentStream = await navigator.mediaDevices.getUserMedia(constraints || defaultConstraints);
            if (this.videoElement) {
                this.videoElement.srcObject = this.currentStream;
                await this.videoElement.play(); // Important to wait for play()
                console.log("Video stream started.");
            } else {
                 throw new Error("Video element not set before playing stream.");
            }
        } catch (error: any) {
            console.error("Camera failed:", error);
            this.streamError = `Camera failed: ${error.message}`;
            this.stopStream(); // Clean up
            throw error; // Re-throw for the orchestrator to handle
        }
    }

    stopStream(): void {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            console.log("Video stream stopped.");
        }
        this.currentStream = null;
        if (this.videoElement) {
            this.videoElement.pause();
            this.videoElement.srcObject = null;
        }
        // Don't nullify videoElement, it might be needed later
    }

    getVideoElement(): HTMLVideoElement | null {
        return this.videoElement;
    }

    getVideoDimensions(): VideoDims | null {
        if (this.videoElement && this.videoElement.videoWidth > 0) {
            return { w: this.videoElement.videoWidth, h: this.videoElement.videoHeight };
        }
        return null;
    }

    isReady(): boolean {
        return !!(this.videoElement && this.currentStream && this.videoElement.readyState >= this.videoElement.HAVE_CURRENT_DATA && !this.videoElement.paused);
    }

    getError(): string | null {
        return this.streamError;
    }
}

// --- 2. ModelLoader Utility ---
class ModelLoader {
    static async loadModel(url: string): Promise<tf.GraphModel> {
        try {
            console.log(`Loading model from ${url}...`);
            const model = await tf.loadGraphModel(url);
            console.log(`Model loaded successfully from ${url}.`);
            return model;
        } catch (error: any) {
            console.error(`Failed to load model from ${url}:`, error);
            throw new Error(`Model load failed (${url}): ${error.message || error}`);
        }
    }
}

// --- 3. YoloProcessor ---
class YoloProcessor {
    constructor(private yoloModel: tf.GraphModel,
                private inputWidth: number,
                private inputHeight: number) {}

    /**
     * Processes a video frame to get detections and prototype masks.
     * NOTE: The returned `maskCoeffs` tensors within RawDetection are slices
     * and MUST be disposed by the caller after use. The protoTensor
     * must also be disposed by the caller.
     */
    async processFrame(frameTensor: tf.Tensor): Promise<{ detections: RawDetection[], protoTensor: tf.Tensor | null }> {
        const memoryInfoStart = await tf.memory(); // Optional: for debugging memory leaks

        let output: tf.Tensor[] | null = null;
        const detections: RawDetection[] = [];
        let protoTensor: tf.Tensor | null = null;
        const tensorsCreated: tf.Tensor[] = []; // Track tensors created here for disposal on error/exit

        try {
            const inputTensor = tf.tidy(() => {
                const resized = tf.image.resizeBilinear(frameTensor, [this.inputHeight, this.inputWidth]);
                const normalized = resized.div(255.0);
                return normalized.expandDims(0).cast('float32');
            });
            tensorsCreated.push(inputTensor); // inputTensor is returned by tidy, track it

            output = await this.yoloModel.executeAsync(inputTensor) as tf.Tensor[];
            if (!output || output.length < 3) {
                console.warn("YOLO output missing expected tensors. Length:", output?.length);
                return { detections: [], protoTensor: null };
            }
            tensorsCreated.push(...output); // Track output tensors

            const detectionsTensor = output[0]; // Shape [1, num_dets, 38]
            protoTensor = output[2];           // Shape [1, 160, 160, 32], Ownership transferred to caller

            // Validate proto tensor shape (important!)
            if (protoTensor.shape.length !== 4 || protoTensor.shape[0] !== 1 ||
                protoTensor.shape[1] !== PROTO_MASK_SIZE || protoTensor.shape[2] !== PROTO_MASK_SIZE ||
                protoTensor.shape[3] !== MASK_COEFF_COUNT) {
                 console.error("Unexpected prototype mask tensor shape:", protoTensor.shape);
                 throw new Error("Incorrect prototype mask shape.");
            }


            const detectionsBatch = await detectionsTensor.data() as Float32Array; // Efficient data access
            const numDets = detectionsTensor.shape[1];
            const detDataLength = detectionsTensor.shape[2]; // e.g., 38

            for (let i = 0; i < numDets; i++) {
                const offset = i * detDataLength;
                const confidence = detectionsBatch[offset + 4];

                if (confidence >= CONFIDENCE_THRESHOLD) {
                    // Slice coefficients tensor - IMPORTANT: Caller must dispose!
                    const maskCoeffs = tf.slice(detectionsTensor, [0, i, DET_BOX_COEFF_INDEX], [1, 1, MASK_COEFF_COUNT]);

                    detections.push({
                        bboxModel: Array.from(detectionsBatch.slice(offset, offset + 4)), // [x1, y1, x2, y2]
                        confidence: confidence,
                        classId: Math.round(detectionsBatch[offset + 5]),
                        maskCoeffs: maskCoeffs, // Transfer ownership to caller
                    });
                }
            }
            // Dispose the large detections tensor now that we've sliced coeffs
            detectionsTensor.dispose();
            // inputTensor and output[1] can be disposed (output[0] was disposed, output[2] is returned)
            inputTensor.dispose();
            if (output[1] && !output[1].isDisposed) output[1].dispose();


        } catch (error) {
            console.error("Error during YOLO processing:", error);
            // Clean up any tensors created if an error occurred before return
            tf.dispose(tensorsCreated); // Dispose all tracked tensors
            tf.dispose(detections.map(d => d.maskCoeffs)); // Dispose any coeffs already sliced
            return { detections: [], protoTensor: null }; // Return empty on error
        }
        // Optional: Check memory usage
        // const memoryInfoEnd = await tf.memory();
        // if(memoryInfoEnd.numBytes > memoryInfoStart.numBytes) {
        //     console.warn("YoloProcessor potential memory leak:", memoryInfoEnd.numBytes - memoryInfoStart.numBytes, "bytes");
        // }

        return { detections, protoTensor }; // protoTensor ownership transferred
    }
}

// --- 4. MaskGenerator ---
class MaskGenerator {
    constructor(private protoMaskSize: number,
                private maskCoeffCount: number,
                private maskCropPaddingFactor: number,
                private maskThreshold: number) {}

    /**
     * Generates a binary mask tensor for a single detection.
     * Disposes intermediate tensors it creates.
     * Does NOT dispose input tensors (protoTensor, maskCoeffs).
     * Returns null if generation fails.
     */
    async generateMask(protoTensor: tf.Tensor, maskCoeffs: tf.Tensor, targetVideoBox: BBox, videoDims: VideoDims): Promise<tf.Tensor2D | null> {
        if (!protoTensor || protoTensor.isDisposed || !maskCoeffs || maskCoeffs.isDisposed) {
            console.warn("generateMask received disposed tensor.");
            return null;
        }

        let finalMask: tf.Tensor2D | null = null;

        try {
            finalMask = tf.tidy(() => { // Use tidy for intermediate tensors
                // 1. Combine protos and coefficients
                const protosReshaped = protoTensor.squeeze(0).reshape([this.protoMaskSize * this.protoMaskSize, this.maskCoeffCount]);
                const coeffsReshaped = maskCoeffs.reshape([this.maskCoeffCount, 1]);
                const maskProto = tf.matMul(protosReshaped, coeffsReshaped);
                const maskReshaped = maskProto.reshape([this.protoMaskSize, this.protoMaskSize]);
                const maskActivated = tf.sigmoid(maskReshaped);

                // 2. Calculate Crop Box (using padding factor)
                const boxWidth = targetVideoBox.x2 - targetVideoBox.x1;
                const boxHeight = targetVideoBox.y2 - targetVideoBox.y1;
                const padX = boxWidth * this.maskCropPaddingFactor;
                const padY = boxHeight * this.maskCropPaddingFactor;
                const cropX1 = Math.max(0, targetVideoBox.x1 - padX);
                const cropY1 = Math.max(0, targetVideoBox.y1 - padY);
                const cropX2 = Math.min(videoDims.w, targetVideoBox.x2 + padX);
                const cropY2 = Math.min(videoDims.h, targetVideoBox.y2 + padY);
                const targetCropW = Math.ceil(cropX2 - cropX1);
                const targetCropH = Math.ceil(cropY2 - cropY1);

                if (targetCropW <= 0 || targetCropH <= 0) return null;

                const normalizedPaddedBbox = [[
                    cropY1 / videoDims.h, cropX1 / videoDims.w,
                    cropY2 / videoDims.h, cropX2 / videoDims.w
                ]];

                // 3. Crop and Resize
                const maskExpanded = maskActivated.expandDims(0).expandDims(-1);
                const maskCroppedResized = tf.image.cropAndResize(
                    maskExpanded, normalizedPaddedBbox, [0], [targetCropH, targetCropW], 'bilinear'
                );

                // 4. Threshold
                // Keep the final mask outside tidy, otherwise it gets disposed
                const mask = maskCroppedResized.squeeze([0, 3]).greater(this.maskThreshold).cast('float32');
                return mask;
            }); // Tidy disposes intermediate tensors like protosReshaped, coeffsReshaped, maskProto, etc.

        } catch (error) {
            console.error("Error generating mask:", error);
            finalMask?.dispose(); // Dispose if created before error
            return null;
        }

        // Detach the mask from the current scope if it's needed later
        // tf.keep(finalMask); // - tf.keep is deprecated. Just returning it should be fine if not nested in another tidy.
        // If finalMask is null, tidy already cleaned up intermediate tensors.
        return finalMask as (tf.Tensor2D | null); // Cast necessary because tidy can return undefined
    }
}

// --- 5. ObjectTracker (Norfair Wrapper) ---
class ObjectTracker {
    private tracker: Tracker;

    constructor(options: TrackerOptions) {
        this.tracker = new Tracker(options);
        console.log("Norfair tracker initialized with options:", options);
    }

    /** Takes detections with *model* coordinate points */
    update(detections: { point: Point }[]): number[] { // Returns array of track IDs or -1
       // Norfair expects points relative to the input coordinate system (model input)
       const norfairDetections = detections.map(d => d.point);
       return this.tracker.update(norfairDetections);
    }

    getTrackedObjects(): TrackedObject[] {
        return this.tracker.trackedObjects;
    }
}

// --- 6. EmbeddingProcessor ---
class EmbeddingProcessor {
    constructor(private embedModel: tf.GraphModel,
                private inputWidth: number,
                private inputHeight: number,
                private cropPaddingFactor: number) {}

    /** Generates an embedding tensor for the given bounding box in video coordinates. */
    async createEmbedding(videoElement: HTMLVideoElement, targetVideoBox: BBox): Promise<tf.Tensor | null> {
        if (!videoElement.videoWidth || !this.embedModel) {
            return null;
        }

        let embedding: tf.Tensor | null = null;
        let clonedEmbedding: tf.Tensor | null = null;

        try {
            const videoWidth = videoElement.videoWidth;
            const videoHeight = videoElement.videoHeight;

            // Calculate padded crop box in video coordinates
            const boxWidth = targetVideoBox.x2 - targetVideoBox.x1;
            const boxHeight = targetVideoBox.y2 - targetVideoBox.y1;
            const padX = boxWidth * this.cropPaddingFactor;
            const padY = boxHeight * this.cropPaddingFactor;
            const cropX1 = Math.max(0, Math.floor(targetVideoBox.x1 - padX));
            const cropY1 = Math.max(0, Math.floor(targetVideoBox.y1 - padY));
            const cropX2 = Math.min(videoWidth, Math.ceil(targetVideoBox.x2 + padX));
            const cropY2 = Math.min(videoHeight, Math.ceil(targetVideoBox.y2 + padY));

            if (cropX2 <= cropX1 || cropY2 <= cropY1) {
                console.warn("Invalid crop dimensions for embedding.");
                return null;
            }

            const boxes = [[cropY1 / videoHeight, cropX1 / videoWidth, cropY2 / videoHeight, cropX2 / videoWidth]];
            const boxIndices = [0];
            const cropSize: [number, number] = [this.inputHeight, this.inputWidth];

            // Use tidy for intermediate tensors related to cropping/preprocessing
            embedding = await tf.tidy(() => {
                const frame = tf.browser.fromPixels(videoElement); // Get frame inside tidy
                const cropped = tf.image.cropAndResize(
                    frame.expandDims(0).toFloat(), boxes, boxIndices, cropSize, 'bilinear'
                );
                const normalized = cropped.div(255.0);
                // Run model - assume executeAsync doesn't dispose its input here
                const result = this.embedModel.execute(normalized) as tf.Tensor; // Use sync execute inside tidy if possible
                return result;
            });
             // tf.tidy disposes frame, cropped, normalized, and potentially the raw model output
             // 'embedding' now holds the result tensor outside the tidy scope.

            if (embedding && embedding.shape && embedding.shape.length > 0) {
                 clonedEmbedding = embedding.clone(); // Clone for the caller
            } else {
                console.warn("Embedding model returned invalid tensor.");
            }

        } catch (error) {
            console.error("Error creating embedding:", error);
        } finally {
             // Dispose the original embedding tensor returned by execute/tidy
             embedding?.dispose();
        }
        return clonedEmbedding; // Return the clone (ownership transferred)
    }
}


// --- 7. CanvasRenderer ---
class CanvasRenderer {
    private visibleCtx: CanvasRenderingContext2D;
    private bufferCanvas: HTMLCanvasElement;
    private bufferCtx: CanvasRenderingContext2D;
    private displayWidth: number = 0;
    private displayHeight: number = 0;
    private videoWidth: number = 0;
    private videoHeight: number = 0;
    public scale: number = 1;
    public offsetX: number = 0;
    public offsetY: number = 0;

    constructor(private visibleCanvas: HTMLCanvasElement) {
        const ctx = visibleCanvas.getContext('2d');
        if (!ctx) throw new Error("Failed to get visible canvas context");
        this.visibleCtx = ctx;

        this.bufferCanvas = document.createElement('canvas');
        const bufCtx = this.bufferCanvas.getContext('2d');
        if (!bufCtx) throw new Error("Failed to get buffer canvas context");
        this.bufferCtx = bufCtx;
    }

    updateDimensions(videoElement: HTMLVideoElement): boolean {
        if (!videoElement.videoWidth || videoElement.videoWidth === 0) return false;

        const newDisplayWidth = videoElement.clientWidth;
        const newDisplayHeight = videoElement.clientHeight;
        const newVideoWidth = videoElement.videoWidth;
        const newVideoHeight = videoElement.videoHeight;

        let changed = false;
        if (this.displayWidth !== newDisplayWidth || this.displayHeight !== newDisplayHeight) {
            this.visibleCanvas.width = newDisplayWidth;
            this.visibleCanvas.height = newDisplayHeight;
            this.bufferCanvas.width = newDisplayWidth;
            this.bufferCanvas.height = newDisplayHeight;
            this.displayWidth = newDisplayWidth;
            this.displayHeight = newDisplayHeight;
            changed = true;
            console.log(`Renderer dimensions updated: ${newDisplayWidth}x${newDisplayHeight}`);
        }

        if (this.videoWidth !== newVideoWidth || this.videoHeight !== newVideoHeight || changed) {
             this.videoWidth = newVideoWidth;
             this.videoHeight = newVideoHeight;
             // Recalculate scale and offset
            const scaleX = this.displayWidth / this.videoWidth;
            const scaleY = this.displayHeight / this.videoHeight;
            this.scale = Math.min(scaleX, scaleY);
            this.offsetX = (this.displayWidth - this.videoWidth * this.scale) / 2;
            this.offsetY = (this.displayHeight - this.videoHeight * this.scale) / 2;
            changed = true;
        }
        return changed;
    }

    // Helper to convert video bbox to canvas bbox
    convertVideoBoxToCanvas(videoBox: BBox): BBoxCanvas {
        return {
            x: videoBox.x1 * this.scale + this.offsetX,
            y: videoBox.y1 * this.scale + this.offsetY,
            w: (videoBox.x2 - videoBox.x1) * this.scale,
            h: (videoBox.y2 - videoBox.y1) * this.scale,
        };
    }

     // Helper to calculate scaled mask dimensions on canvas
    getCanvasMaskDimensions(videoBox: BBox): { w: number; h: number } {
         // This calculation should account for padding if it was used during mask generation
        const boxWidth = videoBox.x2 - videoBox.x1;
        const boxHeight = videoBox.y2 - videoBox.y1;
        const padX = boxWidth * MASK_CROP_PADDING_FACTOR; // Use the same padding factor
        const padY = boxHeight * MASK_CROP_PADDING_FACTOR;
        // Calculate the effective video dimensions *of the mask area*
        const maskVideoWidth = (boxWidth + 2 * padX); // Approximate considering padding was added
        const maskVideoHeight = (boxHeight + 2 * padY); // Clamp this if needed based on actual crop coords
        // This might need refinement based on exact crop coords used in MaskGenerator

        // Return the scaled dimensions
        return {
             w: maskVideoWidth * this.scale,
             h: maskVideoHeight * this.scale,
        };
        // Alternative: Use the mask tensor's actual dimensions and scale them?
        // Needs the mask tensor passed here or calculated earlier. Let's stick to bbox scaling for now.
    }

    clearAll(): void {
        this.visibleCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
        this.bufferCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
    }

    /**
     * Draws a complete frame using the provided objects.
     * Assumes mask tensors will be disposed by the caller afterwards.
     */
    async drawFrame(objectsToDraw: DrawInfo[]): Promise<void> {
        // 1. Clear Buffer
        this.bufferCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);

        // 2. Draw BBoxes and Prepare Mask Promises
        const maskPromises: Promise<void>[] = [];
        for (const obj of objectsToDraw) {
            this.drawBoundingBox(this.bufferCtx, obj.bboxCanvas, obj.color, obj.label);
            if (obj.mask && obj.mask.tensor && !obj.mask.tensor.isDisposed) {
                maskPromises.push(
                    this.drawMask(this.bufferCtx, obj.mask.tensor, obj.color, obj.mask.videoBox)
                );
            }
        }

        // 3. Wait for Masks to be drawn onto buffer
        await Promise.all(maskPromises);

        // 4. Draw Buffer to Visible Canvas
        this.visibleCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
        this.visibleCtx.drawImage(this.bufferCanvas, 0, 0);
    }

    // --- Internal Drawing Helpers ---

    private drawBoundingBox(ctx: CanvasRenderingContext2D, bbox: BBoxCanvas, color: number[], label: string): void {
        ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox.x, bbox.y, bbox.w, bbox.h);

        // Label
        ctx.font = '12px sans-serif';
        ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.8)`;
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 12;
        const padding = 2;
        let textY = bbox.y - padding - 1;
        let backgroundY = textY - textHeight - padding;
        if (backgroundY < 0) {
            textY = bbox.y + bbox.h + textHeight + padding;
            backgroundY = bbox.y + bbox.h + padding;
        }
        ctx.fillRect(bbox.x - 1, backgroundY, textWidth + (padding * 2), textHeight + (padding * 2));
        ctx.fillStyle = `white`;
        ctx.fillText(label, bbox.x + padding - 1, textY);
    }

    /** Draws mask using drawImage for proper scaling */
    private async drawMask(drawCtx: CanvasRenderingContext2D, maskTensor: tf.Tensor2D, color: number[], videoBox: BBox): Promise<void> {
        let tempMaskCanvas: HTMLCanvasElement | null = null;
        const tensorsToDisposeInternally: tf.Tensor[] = [];

        try {
            const [maskTensorHeight, maskTensorWidth] = maskTensor.shape;

             // Calculate the video coordinates of the area the mask tensor covers (includes padding)
            const boxWidth = videoBox.x2 - videoBox.x1;
            const boxHeight = videoBox.y2 - videoBox.y1;
            const padX = boxWidth * MASK_CROP_PADDING_FACTOR;
            const padY = boxHeight * MASK_CROP_PADDING_FACTOR;
            // Top-left corner in video space where the mask begins
            const maskVideoX1 = Math.max(0, videoBox.x1 - padX);
            const maskVideoY1 = Math.max(0, videoBox.y1 - padY);
            // Video dimensions of the mask area (should align with maskTensor dimensions after cropResize)
            // Note: Use the actual tensor dimensions if cropResize might slightly alter aspect ratio?
            // Let's assume maskTensor dimensions accurately represent the intended area after cropResize.
            // We need the effective video width/height *represented by* the mask tensor's pixels.
            // If padding=0, maskVideoWidth = boxWidth. If padding > 0, it's larger.
            // Let's *derive* the video dimensions FROM the tensor shape and the original bbox
            // This assumes cropResize maintained aspect ratio roughly. A better way might be needed.
             // For now, assume the mask tensor maps 1:1 to the padded area calculated.
            const maskVideoWidth = boxWidth + 2 * padX; // Simplified; clamping ignored here
            const maskVideoHeight = boxHeight + 2 * padY;// Simplified

            // Create ImageData
            const colorTensor = tf.tensor1d(color.map(c => c / 255.0));
            tensorsToDisposeInternally.push(colorTensor);
            const maskColored = tf.tidy(() => maskTensor.expandDims(-1).mul(colorTensor.reshape([1, 1, 3])));
            tensorsToDisposeInternally.push(maskColored);
            const alphaChannel = maskTensor.expandDims(-1);
            tensorsToDisposeInternally.push(alphaChannel);
            const maskRgba = tf.tidy(() => tf.concat([maskColored, alphaChannel], -1));
            tensorsToDisposeInternally.push(maskRgba);

            const maskPixelData = await tf.browser.toPixels(maskRgba);
            const imageData = new ImageData(maskPixelData, maskTensorWidth, maskTensorHeight);

            // Create temp canvas
            tempMaskCanvas = document.createElement('canvas');
            tempMaskCanvas.width = maskTensorWidth;
            tempMaskCanvas.height = maskTensorHeight;
            const tempCtx = tempMaskCanvas.getContext('2d');
            if (!tempCtx) throw new Error("Failed to get temp canvas context");
            tempCtx.putImageData(imageData, 0, 0);

            // Calculate destination on target canvas
            const canvasDrawX = maskVideoX1 * this.scale + this.offsetX;
            const canvasDrawY = maskVideoY1 * this.scale + this.offsetY;
            // **Crucially, calculate destination width/height using scale**
            const canvasMaskWidth = maskVideoWidth * this.scale;
            const canvasMaskHeight = maskVideoHeight * this.scale;


            // Draw scaled image
            drawCtx.save();
            drawCtx.globalAlpha = MASK_DRAW_ALPHA;
            if (canvasMaskWidth > 0 && canvasMaskHeight > 0) { // Avoid 0-size draws
                 drawCtx.drawImage(tempMaskCanvas, canvasDrawX, canvasDrawY, canvasMaskWidth, canvasMaskHeight);
            }
            drawCtx.restore();

        } catch (error) {
            console.error("Error drawing scaled mask:", error);
        } finally {
            tf.dispose(tensorsToDisposeInternally); // Dispose tensors created here
            // Temp canvas GC'd
        }
    }
}


// ===========================================================
// Main LitElement Component (Orchestrator)
// ===========================================================

@customElement('video-container')
class VideoContainer extends LitElement {
  static styles = css`
    :host { display: block; width: 100%; height: 100%; overflow: hidden; position: relative; background-color: #222; }
    video { display: block; width: 100%; height: 100%; object-fit: contain; background-color: #000; }
    canvas#overlay-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 5; }
    .status { position: absolute; bottom: 0; left: 0; right: 0; background-color: rgba(0, 0, 0, 0.7); color: white; font-size: 0.9em; padding: 5px 10px; text-align: center; z-index: 10; }
  `;

  // --- Element Refs ---
  @query('#video') private videoElement!: HTMLVideoElement;
  @query('#overlay-canvas') private canvasElement!: HTMLCanvasElement;

  // --- State ---
  @state() private inferenceStatus: string = 'Initializing...';
  @state() private modelsReady: boolean = false;
  @state() private streamReady: boolean = false;

  // High-level state for tracked objects
  private objectEmbeddingStatus = new Map<number, ObjectEmbeddingInfo>();

  // --- Service Instances ---
  private videoSourceManager!: VideoSourceManager;
  private yoloProcessor!: YoloProcessor;
  private maskGenerator!: MaskGenerator;
  private objectTracker!: ObjectTracker;
  private embeddingProcessor!: EmbeddingProcessor;
  private canvasRenderer!: CanvasRenderer;
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
    console.log("VideoContainer connected");
    this.videoSourceManager = new VideoSourceManager();
    // Services requiring models will be fully initialized after model loading
    this.objectTracker = new ObjectTracker({
        distanceThreshold: 100, hitInertiaMin: 0, hitInertiaMax: 15, initDelay: 2,
    });
    this.maskGenerator = new MaskGenerator(
        PROTO_MASK_SIZE, MASK_COEFF_COUNT, MASK_CROP_PADDING_FACTOR, MASK_THRESHOLD
    );

    tf.ready().then(() => {
        this.loadModels(); // Don't await here, let it run in background
    }).catch(tfError => {
         this.handleFatalError("TF.js initialization failed", tfError);
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    console.log("VideoContainer disconnected");
    this.stopLoops();
    this.videoSourceManager?.stopStream();
    this.resizeObserver?.disconnect();
    // Dispose models
    this.yoloModel?.dispose();
    this.embedModel?.dispose();
    // Dispose stored embeddings
    this.objectEmbeddingStatus.forEach(info => info.embedding?.dispose());
    this.objectEmbeddingStatus.clear();
    // Clear canvases (renderer might be null if error occurred early)
    this.canvasRenderer?.clearAll();
    console.log("Cleanup complete.");
  }

  protected firstUpdated() {
    console.log("VideoContainer firstUpdated");
    if (!this.videoElement || !this.canvasElement) {
        this.handleFatalError("Initialization Error: Video or Canvas element missing.");
        return;
    }
    try {
        this.canvasRenderer = new CanvasRenderer(this.canvasElement);
        this.resizeObserver = new ResizeObserver(() => {
             if(this.videoSourceManager.getVideoElement()) {
                 this.canvasRenderer?.updateDimensions(this.videoSourceManager.getVideoElement()!);
             }
         });
        this.resizeObserver.observe(this.videoElement); // Observe video element size

        this.videoElement.addEventListener('loadedmetadata', this.handleVideoMetadataLoaded);
        this.videoElement.addEventListener('error', (e) => this.handleFatalError('Video element error.', e));
        // Attempt to start stream immediately if permissions granted previously
        this.startVideoStream();
    } catch(error) {
        this.handleFatalError("Error during firstUpdated setup", error);
    }
  }

  // --- Orchestration ---

  private async loadModels() {
    this.inferenceStatus = 'Loading models...';
    this.modelsReady = false;
    try {
        const [yolo, embed] = await Promise.all([
            ModelLoader.loadModel(MODEL_URL),
            ModelLoader.loadModel(EMBED_URL)
        ]);
        this.yoloModel = yolo;
        this.embedModel = embed;

        // Initialize services that depend on models
        this.yoloProcessor = new YoloProcessor(this.yoloModel, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
        this.embeddingProcessor = new EmbeddingProcessor(this.embedModel, EMBED_INPUT_WIDTH, EMBED_INPUT_HEIGHT, EMBEDDING_CROP_PADDING_FACTOR);

        this.modelsReady = true;
        this.inferenceStatus = 'Models loaded.';
        this.checkAndStartLoops(); // Attempt to start loops if stream is also ready

    } catch (error: any) {
        this.handleFatalError("Model loading failed", error);
        this.modelsReady = false; // Ensure loops don't start
    }
  }

  private async startVideoStream() {
      if(!this.videoElement) return;
      this.inferenceStatus = 'Requesting camera...';
      this.streamReady = false;
      try {
         await this.videoSourceManager.startStream(this.videoElement);
         this.streamReady = true;
         this.inferenceStatus = 'Video stream started.';
         this.canvasRenderer?.updateDimensions(this.videoElement); // Initial size update
         this.checkAndStartLoops(); // Attempt to start loops if models are also ready
      } catch(error: any) {
          // Error already logged by VideoSourceManager
          this.inferenceStatus = this.videoSourceManager.getError() || 'Camera access failed.';
          this.streamReady = false;
      }
  }

  private handleVideoMetadataLoaded = () => {
      console.log("Video metadata loaded.");
      if (this.canvasRenderer && this.videoElement) {
          this.canvasRenderer.updateDimensions(this.videoElement); // Ensure correct scale/offset
      }
      this.checkAndStartLoops(); // May now be ready
  }

  private checkAndStartLoops() {
      if (this.modelsReady && this.streamReady && this.videoSourceManager.isReady()) {
          console.log("Models and stream ready, starting loops.");
          this.inferenceStatus = 'Starting inference...';
          this.startInferenceLoop();
          this.startEmbeddingLoop();
      } else {
           console.log(`Loops not started: Models=${this.modelsReady}, Stream=${this.streamReady}, VideoState=${this.videoSourceManager.isReady()}`);
           // Update status based on what's missing
           if (!this.modelsReady) this.inferenceStatus = "Waiting for models...";
           else if (!this.streamReady) this.inferenceStatus = "Waiting for camera...";
           else if (!this.videoSourceManager.isReady()) this.inferenceStatus = "Waiting for video playback...";
      }
  }

   private startInferenceLoop() {
    if (this.inferenceLoopId === null && this.modelsReady && this.streamReady) {
        console.log("Starting main inference loop");
        this.isInferencing = false;
        this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
    }
   }

   private startEmbeddingLoop() {
     if (this.embeddingLoopTimeoutId === null && this.modelsReady && this.streamReady) {
        console.log("Starting embedding loop");
        this.isEmbedding = false;
        // Use setTimeout for less frequent embedding checks
        this.embeddingLoopTimeoutId = window.setTimeout(() => this.runEmbeddingLoop(), EMBEDDING_LOOP_INTERVAL_MS);
     }
   }

   private stopLoops() {
       if (this.inferenceLoopId !== null) {
           cancelAnimationFrame(this.inferenceLoopId);
           this.inferenceLoopId = null;
           console.log("Inference loop stopped.");
       }
        if (this.embeddingLoopTimeoutId !== null) {
           clearTimeout(this.embeddingLoopTimeoutId);
           this.embeddingLoopTimeoutId = null;
           console.log("Embedding loop stopped.");
        }
        this.isInferencing = false;
        this.isEmbedding = false;
   }

    private async runInference() {
        if (this.isInferencing) { return; } // Prevent re-entry

        // Check prerequisites for this frame
        if (!this.modelsReady || !this.streamReady || !this.videoSourceManager.isReady() || !this.yoloProcessor || !this.canvasRenderer || !this.objectTracker || !this.maskGenerator) {
            this.inferenceLoopId = requestAnimationFrame(() => this.runInference()); // Try again next frame
            return;
        }

        this.isInferencing = true;
        const frameStart = performance.now(); // For FPS calculation (optional)
        const videoElement = this.videoSourceManager.getVideoElement()!;
        const videoDims = this.videoSourceManager.getVideoDimensions()!;

        // Ensure renderer dimensions are up-to-date
        this.canvasRenderer.updateDimensions(videoElement);

        let frameTensor: tf.Tensor | null = null;
        let yoloOutput: { detections: RawDetection[], protoTensor: tf.Tensor | null } | null = null;
        const generatedMasks: Map<number, tf.Tensor2D> = new Map(); // Store generated masks for this frame <norfairIndex, maskTensor>
        const objectsToDraw: DrawInfo[] = [];
        let trackingResults: number[] = []; // Track IDs

        try {
            // 1. Get Frame
            frameTensor = tf.browser.fromPixels(videoElement);

            // 2. Run YOLO
            yoloOutput = await this.yoloProcessor.processFrame(frameTensor);

            // 3. Prepare for Tracker
            const trackerInputs = yoloOutput.detections.map(det => ({
                point: [ (det.bboxModel[0] + det.bboxModel[2]) / 2, (det.bboxModel[1] + det.bboxModel[3]) / 2 ] as Point
            }));

            // 4. Update Tracker
            trackingResults = this.objectTracker.update(trackerInputs);

            // 5. Process Tracked Detections (Generate Masks, Prepare for Drawing)
            const maskGenPromises: Promise<void>[] = [];

            for (let i = 0; i < trackerInputs.length; i++) {
                 const trackId = trackingResults[i];
                 const rawDet = yoloOutput.detections[i]; // Corresponding raw detection

                 // Only process/draw if it's being tracked (or newly detected for init phase)
                 // Note: Norfair might assign -1 initially. Draw these too? Yes.
                 // if (trackId !== -1) { // Or remove this check to draw all detections

                    const modelBbox = rawDet.bboxModel; // Bbox in model coords
                    const videoBbox = this.scaleModelBboxToVideo(modelBbox); // Convert to video coords

                    if (videoBbox && yoloOutput.protoTensor) {
                         // Update last known bbox *before* mask generation uses it for cropping info
                         if (trackId !== -1) {
                             this.updateLastKnownBbox(trackId, videoBbox);
                         }

                         // Generate mask asynchronously
                         const maskPromise = this.maskGenerator.generateMask(
                             yoloOutput.protoTensor, rawDet.maskCoeffs, videoBbox, videoDims
                         ).then(maskTensor => {
                             if (maskTensor) {
                                 generatedMasks.set(i, maskTensor); // Store mask keyed by original detection index
                             }
                         });
                         maskGenPromises.push(maskPromise);

                         // Prepare basic DrawInfo (bbox, label) - mask added later
                         const canvasBbox = this.canvasRenderer.convertVideoBoxToCanvas(videoBbox);
                         const label = `${trackId === -1 ? 'Init' : `ID: ${trackId}`} (${rawDet.confidence.toFixed(2)})`;
                         const color = COLORS[trackId >= 0 ? trackId % COLORS.length : COLORS.length - 1];

                         objectsToDraw.push({
                              id: trackId,
                              color: color,
                              bboxCanvas: canvasBbox,
                              label: label,
                              // Mask will be added after promises resolve
                         });
                    }
                 // } // End trackId check (optional)
            } // End loop through detections

            // Wait for all mask generations to complete
            await Promise.all(maskGenPromises);

            // Add generated masks to the DrawInfo structures
            objectsToDraw.forEach((drawInfo, index) => {
                 // Find the original detection index corresponding to this drawInfo object
                 // This assumes objectsToDraw is created in the same order as detections
                 const originalDetIndex = trackerInputs.findIndex((_, idx) => trackingResults[idx] === drawInfo.id); // Find first match by ID - imperfect if multiple match
                 // A better way: Map trackId back to original detection index i more robustly if needed.
                 // For now, let's assume order is preserved or use index directly if not filtering by trackId above.
                 const correspondingMask = generatedMasks.get(index); // Use direct index assuming order
                 if (correspondingMask) {
                     // We need the original videoBox that this mask corresponds to
                     const rawDet = yoloOutput.detections[index];
                     const videoBbox = this.scaleModelBboxToVideo(rawDet.bboxModel);
                      if(videoBbox){
                           drawInfo.mask = { tensor: correspondingMask, videoBox: videoBbox };
                      } else {
                           correspondingMask.dispose(); // Dispose if bbox invalid
                      }
                 }
            });

             // 6. Update Global Object Status (creation times etc.)
             this.updateObjectStatus(this.objectTracker.getTrackedObjects(), frameStart);


            // 7. Render Frame
            await this.canvasRenderer.drawFrame(objectsToDraw);


        } catch (error) {
            console.error("Error in inference loop:", error);
            this.inferenceStatus = `Error: ${error instanceof Error ? error.message : error}`;
            // Attempt to clear renderer on error?
            this.canvasRenderer?.clearAll();

        } finally {
            // 8. Cleanup Tensors for this frame
            frameTensor?.dispose();
            // Dispose raw mask coefficient tensors from YoloProcessor output
            yoloOutput?.detections.forEach(det => det.maskCoeffs.dispose());
            // Dispose the prototype tensor from YoloProcessor output
            yoloOutput?.protoTensor?.dispose();
            // Dispose generated binary mask tensors *after* rendering
            generatedMasks.forEach(mask => mask.dispose());

            this.isInferencing = false;
            const frameEnd = performance.now();
            const fps = 1000 / (frameEnd - frameStart);
            // Update status only if no error occurred
            if (!this.inferenceStatus.startsWith("Error:")) {
                 this.inferenceStatus = `Tracking ${this.objectTracker.getTrackedObjects().filter(o=>o.id!==-1).length} objs | FPS: ${fps.toFixed(1)}`;
            }

            // Schedule next frame
            if (this.inferenceLoopId !== null) {
                this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
            }
        }
    }


    private async runEmbeddingLoop() {
        if (this.isEmbedding) { return; } // Prevent re-entry

        if (!this.modelsReady || !this.streamReady || !this.videoSourceManager.isReady() || !this.embeddingProcessor) {
             // Reschedule if prerequisites not met
             this.embeddingLoopTimeoutId = window.setTimeout(() => this.runEmbeddingLoop(), EMBEDDING_LOOP_INTERVAL_MS);
             return;
        }

        this.isEmbedding = true;

        try {
             const objectToEmbed = this.selectObjectForEmbedding(); // Find oldest unembedded or oldest embedded
             if (objectToEmbed && this.objectEmbeddingStatus.has(objectToEmbed.id)) {
                  const objectInfo = this.objectEmbeddingStatus.get(objectToEmbed.id)!;
                  const videoElement = this.videoSourceManager.getVideoElement()!;
                  const bboxVideo = objectInfo.lastKnownBboxVideo; // Get last known position

                  if (bboxVideo) {
                       // console.log(`Attempting embedding for ID: ${objectToEmbed.id}`);
                       const newEmbedding = await this.embeddingProcessor.createEmbedding(videoElement, bboxVideo);

                       if (newEmbedding) {
                            // Dispose old embedding and store new one
                            objectInfo.embedding?.dispose();
                            objectInfo.embedding = newEmbedding; // Takes ownership of the cloned tensor
                            objectInfo.hasBeenEmbedded = true;
                            objectInfo.lastEmbeddingTime = Date.now();
                            // console.log(`Embedding stored for ID: ${objectToEmbed.id}`);

                            // Trigger DB Fetch (async, don't necessarily await)
                            this.fetchDataFromVectorDB(newEmbedding, objectInfo.id);
                       }
                  }
             }
        } catch(error) {
             console.error("Error in embedding loop:", error);
        } finally {
             this.isEmbedding = false;
             // Schedule next run
             if (this.embeddingLoopTimeoutId !== null) {
                  this.embeddingLoopTimeoutId = window.setTimeout(() => this.runEmbeddingLoop(), EMBEDDING_LOOP_INTERVAL_MS);
             }
        }
    }

  // --- Helper Methods ---

  // Scales MODEL coordinates (0-INPUT_W/H) to VIDEO coordinates (0-videoW/H)
  private scaleModelBboxToVideo(bboxModel: number[]): BBox | null {
     const dims = this.videoSourceManager.getVideoDimensions();
     if (!dims) return null;
     if(bboxModel.length < 4) return null;
     const [x1_m, y1_m, x2_m, y2_m] = bboxModel;
     return {
         x1: (x1_m / MODEL_INPUT_WIDTH) * dims.w,
         y1: (y1_m / MODEL_INPUT_HEIGHT) * dims.h,
         x2: (x2_m / MODEL_INPUT_WIDTH) * dims.w,
         y2: (y2_m / MODEL_INPUT_HEIGHT) * dims.h,
     };
  }

  // Manages adding/removing objects from the state map and disposing embeddings
  private updateObjectStatus(currentTrackedObjects: TrackedObject[], timestamp: number) {
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
        // console.log(`Removing object ID ${id}`); // Less verbose
        info?.embedding?.dispose(); // Dispose stored embedding
        this.objectEmbeddingStatus.delete(id);
    });
    // Add new objects
    currentTrackedObjects.forEach(obj => {
        if (obj.id !== -1 && !this.objectEmbeddingStatus.has(obj.id)) {
            // console.log(`Adding object ID ${obj.id}`); // Less verbose
            this.objectEmbeddingStatus.set(obj.id, {
                id: obj.id, hasBeenEmbedded: false, lastEmbeddingTime: null,
                creationTime: timestamp, embedding: null, lastKnownBboxVideo: null,
            });
        }
    });
 }

 // Updates the last known video bbox for an active object
 private updateLastKnownBbox(trackId: number, bboxVideo: BBox): void {
     const info = this.objectEmbeddingStatus.get(trackId);
     if (info) {
         info.lastKnownBboxVideo = bboxVideo;
     }
 }

 // Selects candidate for embedding
 private selectObjectForEmbedding(): TrackedObject | null {
     const activeTracked = this.objectTracker?.getTrackedObjects().filter(obj => obj.id !== -1) ?? [];
     if (activeTracked.length === 0) return null;

     const candidatesInfo = activeTracked
         .map(obj => this.objectEmbeddingStatus.get(obj.id))
         .filter((info): info is ObjectEmbeddingInfo => info !== undefined && info.lastKnownBboxVideo !== null); // Must have bbox

     // Priority 1: Unembedded
     const unembedded = candidatesInfo.filter(info => !info.hasBeenEmbedded).sort((a, b) => a.creationTime - b.creationTime);
     if (unembedded.length > 0) {
         return activeTracked.find(obj => obj.id === unembedded[0].id) || null;
     }
     // Priority 2: Embedded, oldest first
     const embedded = candidatesInfo.filter(info => info.hasBeenEmbedded && info.lastEmbeddingTime !== null).sort((a, b) => a.lastEmbeddingTime! - b.lastEmbeddingTime!);
      if (embedded.length > 0) {
         return activeTracked.find(obj => obj.id === embedded[0].id) || null;
     }
      // Fallback: Any embedded object if none have timestamps (less likely)
      const fallback = candidatesInfo.filter(info => info.hasBeenEmbedded);
        if(fallback.length > 0) {
             return activeTracked.find(obj => obj.id === fallback[0].id) || null;
        }

     return null; // No suitable candidate
 }

  // Placeholder for DB interaction
  private async fetchDataFromVectorDB(embedding: tf.Tensor, objectId: number): Promise<void> {
    // console.log(`DB Fetch for ID ${objectId}, Embedding Shape: ${embedding?.shape}`); // Less verbose
    await new Promise(resolve => setTimeout(resolve, 50)); // Simulate async
    // In a real app: send embedding data (e.g., await embedding.array()) to backend API
  }

   private handleFatalError(message: string, error?: any) {
      console.error("FATAL ERROR:", message, error);
      this.inferenceStatus = `FATAL: ${message}`;
      this.stopLoops(); // Stop processing loops
      this.videoSourceManager?.stopStream(); // Stop video stream
      // Models are disposed in disconnectedCallback
      this.modelsReady = false; // Prevent loop restarts
      this.streamReady = false;
   }


  // --- Render ---
  render() {
    return html`
      <video id="video" muted playsinline></video>
      <canvas id="overlay-canvas"></canvas>
      <div class="status">${this.inferenceStatus}</div>
    `;
  }
}

// ===========================================================
// Entry Point / Main Function
// ===========================================================

export function main() {
  // Simple check if custom element already exists
  if (!customElements.get('video-container')) {
      // Define the custom element if it doesn't exist.
      // This might not be strictly necessary if using the decorator,
      // but ensures it's defined before creating instance.
      customElements.define('video-container', VideoContainer);
      console.log("Defined video-container element.");
  } else {
       console.log("video-container element already defined.");
  }

  // Remove existing container if present
  const existingContainer = document.querySelector('app-container video-container');
  if (existingContainer) {
      existingContainer.remove();
  }
   const existingApp = document.querySelector('app-container');
    if (existingApp) {
        existingApp.remove();
    }


  // Create and append the main app container element
  // We might not need app-container if video-container is the root now?
  // Let's assume we still want app-container
   const appContainer = document.createElement('div'); // Use div or define app-container element
   appContainer.style.width = '100%';
   appContainer.style.height = '100%';
   const videoContainer = document.createElement('video-container');
   appContainer.appendChild(videoContainer);
   document.body.appendChild(appContainer);

   // If app-container was a Lit element:
   // const appContainer = document.createElement('app-container');
   // document.body.appendChild(appContainer);
   // Wait for app-container to render and then append video-container? Or render video-container inside app-container's template.
   // For simplicity now, just creating video-container directly.
   // const videoContainer = document.createElement('video-container');
   // document.body.appendChild(videoContainer);


}

// Example: Call main() on script load if desired
// main();
