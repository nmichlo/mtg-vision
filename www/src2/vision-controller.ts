/**
 * Vision Controller
 *
 * Core controller that manages the computer vision pipeline for the MTG Vision application.
 * This class coordinates all the individual services and processes required for:
 * - Camera stream management
 * - TensorFlow.js model loading and inference
 * - Object detection with YOLOv11
 * - Segmentation masks extraction
 * - Object tracking with Norfair
 * - Embedding generation for card identification
 * - Vector index search for card matching
 * - Drawing and rendering of results
 *
 * It decouples the business logic from the UI, allowing for cleaner separation of concerns.
 * Communicates with the UI via callbacks for status updates and card preview generation.
 */

import * as tf from "@tensorflow/tfjs";
import {
  BBox,
  DRAWING_CONFIG,
  DrawInfo,
  EMBEDDING_CONFIG,
  EMBEDDING_LOOP_INTERVAL_MS,
  TRACKER_CONFIG,
  VideoDims,
  YOLO_CONFIG
} from "./types";
import { VideoSourceManager } from "./service-video";
import { ModelLoader, YoloProcessor } from "./service-model";
import { ObjectTracker, TrackedObjectState } from "./service-tracker";
import { EmbeddingProcessor, fetchCardDataFromScryfall } from "./service-embedding";
import { CanvasRenderer } from "./service-canvas";
import { generateMask } from "./util-tensor";
import { getIndexState, searchIndex } from "./util-index";

/**
 * Manages the vision processing flow including model loading,
 * inference, tracking, and embedding
 */
export class VisionController {
  // --- State ---
  private modelsReady: boolean = false;
  private streamReady: boolean = false;

  // --- Services ---
  private videoSourceManager: VideoSourceManager;
  private yoloProcessor?: YoloProcessor;
  private objectTracker: ObjectTracker;
  private embeddingProcessor?: EmbeddingProcessor;
  private canvasRenderer?: CanvasRenderer;
  private trackedObjectState: TrackedObjectState;
  private yoloModel: tf.GraphModel | null = null;
  private embedModel: tf.GraphModel | null = null;

  // --- Loop Control ---
  private inferenceLoopId: number | null = null;
  private embeddingLoopTimeoutId: number | null = null;
  private isInferencing: boolean = false;
  private isEmbedding: boolean = false;

  // --- Events ---
  private statusCallback: (status: string) => void;
  private cardPreviewCallback?: (id: number, image: string) => void;

  constructor(
    statusCallback: (status: string) => void,
    cardPreviewCallback?: (id: number, image: string) => void
  ) {
    this.videoSourceManager = new VideoSourceManager();
    this.objectTracker = new ObjectTracker(TRACKER_CONFIG);
    this.trackedObjectState = new TrackedObjectState();
    this.statusCallback = statusCallback;
    this.cardPreviewCallback = cardPreviewCallback;
  }

  // --- Lifecycle Methods ---

  /**
   * Initializes the vision controller
   */
  async initialize(
    videoElement: HTMLVideoElement,
    canvasElement: HTMLCanvasElement
  ): Promise<void> {
    try {
      // Initialize canvas renderer
      this.canvasRenderer = new CanvasRenderer(canvasElement, DRAWING_CONFIG);

      // Setup video event listeners
      videoElement.addEventListener("loadedmetadata", () => {
        if (this.canvasRenderer) {
          const vid = this.videoSourceManager.getVideoElement();
          if (vid) this.canvasRenderer.updateDimensions(vid);
        }
        this.checkAndStartLoops();
      });

      // Initialize TensorFlow.js
      await tf.ready();

      // Load models
      await this.loadModels();

      // Start video stream
      await this.startVideoStream(videoElement);
    } catch (error) {
      this.handleFatalError("Initialization failed", error);
    }
  }

  /**
   * Cleans up all resources
   */
  dispose(): void {
    this.stopLoops();
    this.videoSourceManager.stopStream();
    this.yoloModel?.dispose();
    this.embedModel?.dispose();
    this.trackedObjectState.disposeAllEmbeddings();
    this.canvasRenderer?.clearAll();
  }

  // --- Model Loading ---

  /**
   * Loads ML models for detection and embedding
   */
  private async loadModels(): Promise<void> {
    this.statusCallback("Loading models...");
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
      this.statusCallback("Models loaded.");
      this.checkAndStartLoops();
    } catch (error: any) {
      this.handleFatalError("Model load failed", error);
      this.modelsReady = false;
    }
  }

  // --- Video Stream Management ---

  /**
   * Starts the camera video stream
   */
  private async startVideoStream(videoElement: HTMLVideoElement): Promise<void> {
    if (!videoElement) return;
    this.statusCallback("Requesting camera...");
    this.streamReady = false;
    try {
      await this.videoSourceManager.startStream(videoElement);
      this.streamReady = true;
      this.statusCallback("Video stream started.");
      this.canvasRenderer?.updateDimensions(videoElement);
      this.checkAndStartLoops();
    } catch (error: any) {
      this.statusCallback(
        this.videoSourceManager.getError() || "Camera failed."
      );
      this.streamReady = false;
    }
  }

  // --- Processing Loops ---

  /**
   * Checks if preconditions are met and starts processing loops
   */
  private checkAndStartLoops(): void {
    if (
      this.modelsReady &&
      this.streamReady &&
      this.videoSourceManager.isReady()
    ) {
      this.statusCallback("Starting...");
      this.startInferenceLoop();
      this.startEmbeddingLoop();
    } else {
      if (!this.modelsReady) this.statusCallback("Waiting for models...");
      else if (!this.streamReady) this.statusCallback("Waiting for camera...");
      else if (!this.videoSourceManager.isReady())
        this.statusCallback("Waiting for video...");
    }
  }

  /**
   * Starts the main inference loop for object detection
   */
  private startInferenceLoop(): void {
    if (
      this.inferenceLoopId === null &&
      this.modelsReady &&
      this.streamReady
    ) {
      this.isInferencing = false;
      this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
    }
  }

  /**
   * Starts the embedding loop for card identification
   */
  private startEmbeddingLoop(): void {
    if (
      this.embeddingLoopTimeoutId === null &&
      this.modelsReady &&
      this.streamReady
    ) {
      this.isEmbedding = false;
      this.embeddingLoopTimeoutId = window.setTimeout(
        () => this.runEmbeddingLoop(),
        EMBEDDING_LOOP_INTERVAL_MS,
      );
    }
  }

  /**
   * Stops all processing loops
   */
  private stopLoops(): void {
    if (this.inferenceLoopId !== null) {
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

  /**
   * Runs a single iteration of the inference loop
   */
  private async runInference(): Promise<void> {
    if (this.isInferencing) {
      return;
    }
    if (!this._checkInferencePrerequisites()) {
      this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
      return;
    }

    this.isInferencing = true;
    const frameStart = performance.now();
    const videoElement = this.videoSourceManager.getVideoElement()!;
    const videoDims = this.videoSourceManager.getVideoDimensions()!;
    this.canvasRenderer!.updateDimensions(videoElement);

    let yoloOutput: {
      detections: any[];
      protoTensor: tf.Tensor | null;
    } | null = null;
    let generatedMasks: Map<number, tf.Tensor2D> | null = null;

    try {
      yoloOutput = await this._getFrameAndYoloOutput(videoElement);
      if (!yoloOutput || !yoloOutput.protoTensor)
        throw new Error("YOLO processing failed.");

      const trackingResults = this._updateTrackerAndGetResults(yoloOutput);

      // Get currently tracked IDs
      const trackedIds = new Set(trackingResults.filter(id => id !== -1));

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
      );

      const objectsToDraw = this._prepareDrawData(
        yoloOutput,
        trackingResults,
        generatedMasks,
      );

      await this.canvasRenderer!.drawFrame(objectsToDraw);

      // Update status
      const fps = 1000 / (performance.now() - frameStart);
      const trackedCount = this.objectTracker
        .getTrackedObjects()
        .filter((o) => o.id !== -1).length;
      this.statusCallback(`Tracking ${trackedCount} | FPS: ${fps.toFixed(1)}`);
    } catch (error) {
      console.error("Error in inference loop:", error);
      this.statusCallback(`Error: ${error instanceof Error ? error.message : error}`);
      this.canvasRenderer?.clearAll();
    } finally {
      this._cleanupInferenceResources(yoloOutput, generatedMasks); // Cleanup inter-stage tensors
      this.isInferencing = false;
      if (this.inferenceLoopId !== null) {
        this.inferenceLoopId = requestAnimationFrame(() => this.runInference());
      }
    }
  }

  // --- Inference Helper Methods ---

  private _checkInferencePrerequisites(): boolean {
    return Boolean(
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
  ): Promise<{ detections: any[]; protoTensor: tf.Tensor | null }> {
    let frameTensor: tf.Tensor | null = null;
    try {
      frameTensor = tf.browser.fromPixels(videoElement);
      const output = await this.yoloProcessor!.processFrame(frameTensor);
      return output;
    } finally {
      frameTensor?.dispose();
    }
  }

  private _updateTrackerAndGetResults(yoloOutput: {
    detections: any[];
  }): number[] {
    const trackerInputs = yoloOutput.detections.map((det) => ({
      point: [
        (det.bboxModel[0] + det.bboxModel[2]) / 2,
        (det.bboxModel[1] + det.bboxModel[3]) / 2,
      ] as [number, number],
    }));
    return this.objectTracker.update(trackerInputs);
  }

  // Scales MODEL coordinates (0-INPUT_W/H) to VIDEO coordinates (0-videoW/H)
  private scaleModelBboxToVideo(bboxModel: number[]): BBox | null {
    const dims = this.videoSourceManager.getVideoDimensions();
    if (!dims || bboxModel.length < 4) return null;
    const [x1_m, y1_m, x2_m, y2_m] = bboxModel;
    return {
      x1: (x1_m / YOLO_CONFIG.inputWidth) * dims.w,
      y1: (y1_m / YOLO_CONFIG.inputHeight) * dims.h,
      x2: (x2_m / YOLO_CONFIG.inputWidth) * dims.w,
      y2: (y2_m / YOLO_CONFIG.inputHeight) * dims.h,
    };
  }

  // Uses the state manager to update bboxes
  private _updateTrackedObjectBBoxes(
    yoloOutput: { detections: any[] },
    trackingResults: number[],
  ): void {
    yoloOutput.detections.forEach((rawDet, index) => {
      const trackId = trackingResults[index];
      if (trackId !== -1) {
        const videoBbox = this.scaleModelBboxToVideo(rawDet.bboxModel);
        this.trackedObjectState.updateLastKnownBbox(trackId, videoBbox);
      }
    });
  }

  private async _generateMasksForFrame(
    yoloOutput: { detections: any[]; protoTensor: tf.Tensor | null },
    trackingResults: number[],
    videoDims: VideoDims,
    maskGenConfig: any,
  ): Promise<Map<number, tf.Tensor2D>> {
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
    yoloOutput: { detections: any[] },
    trackingResults: number[],
    generatedMasks: Map<number, tf.Tensor2D> | null,
  ): DrawInfo[] {
    const objectsToDraw: DrawInfo[] = [];
    yoloOutput.detections.forEach((rawDet, index) => {
      const trackId = trackingResults[index];
      const videoBbox = this.scaleModelBboxToVideo(rawDet.bboxModel);
      if (videoBbox) {
        const canvasBbox = this.canvasRenderer!.convertVideoBoxToCanvas(videoBbox);
        const objectInfo = this.trackedObjectState.get(trackId);
        const cardName = objectInfo?.cardData?.name || '';
        const label = `${trackId === -1 ? "Init" : `ID: ${trackId}`}${cardName ? ` - ${cardName}` : ''} (${rawDet.confidence.toFixed(2)})`;
        const color = DRAWING_CONFIG.colors[
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
      detections: any[];
      protoTensor: tf.Tensor | null;
    } | null,
    generatedMasks: Map<number, tf.Tensor2D> | null,
  ): void {
    yoloOutput?.detections.forEach((det) => det.maskCoeffs.dispose());
    yoloOutput?.protoTensor?.dispose();
    generatedMasks?.forEach((mask) => mask.dispose());
  }

  // --- Embedding Loop ---

  /**
   * Runs a single iteration of the embedding loop
   */
  private async runEmbeddingLoop(): Promise<void> {
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

        // Capture and display the card image if callback is provided
        if (this.cardPreviewCallback) {
          const cardImage = await this.captureCardImage(videoElement, bboxVideo);
          this.cardPreviewCallback(id, cardImage);
        }

        const newEmbedding = await this.embeddingProcessor.createEmbedding(
          videoElement,
          bboxVideo,
        );
        if (newEmbedding) {
          this.trackedObjectState.updateEmbedding(id, newEmbedding);
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

  // --- Helper Methods ---

  /**
   * Searches for matching cards in the vector DB
   */
  private async fetchDataFromVectorDB(embedding: tf.Tensor, objectId: number): Promise<string | null> {
    const { indexProcessVec } = getIndexState();

    if (!indexProcessVec) {
      console.warn("Index not loaded for search");
      return null;
    }

    const embArray = Array.from(await embedding.data() as Float32Array);
    const results = await searchIndex(embArray, 1);
    const result = results?.[0];

    // Fetch card data from Scryfall
    if (result?.id) {
      const cardData = await fetchCardDataFromScryfall(result.id);
      if (cardData) {
        this.trackedObjectState.updateCardData(objectId, cardData);
      }
    }

    return result?.id || null;
  }

  /**
   * Handles fatal errors by cleaning up resources
   */
  private handleFatalError(message: string, error?: any): void {
    console.error(
      "FATAL ERROR:",
      message,
      error,
    );
    this.statusCallback(`FATAL: ${message}`);
    this.stopLoops();
    this.videoSourceManager.stopStream();
    this.modelsReady = false;
    this.streamReady = false;
    this.canvasRenderer?.clearAll();
  }

  /**
   * Captures an image of a card from the video frame
   */
  private async captureCardImage(videoElement: HTMLVideoElement, bbox: BBox): Promise<string> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    // Set canvas size to match the bbox
    canvas.width = bbox.x2 - bbox.x1;
    canvas.height = bbox.y2 - bbox.y1;

    // Draw the video frame to the canvas, cropping to the bbox
    ctx.drawImage(
      videoElement,
      bbox.x1, bbox.y1, canvas.width, canvas.height, // source
      0, 0, canvas.width, canvas.height // destination
    );

    return canvas.toDataURL('image/jpeg', 0.8);
  }

  /**
   * Gets the TrackedObjectState for accessing object info
   */
  getTrackedObjectState(): TrackedObjectState {
    return this.trackedObjectState;
  }
}
