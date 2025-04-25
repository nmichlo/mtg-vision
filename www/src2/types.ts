/**
 * Type Definitions
 *
 * Central repository for all shared types, interfaces, and constants used
 * throughout the MTG Vision application. This file contains:
 * - Interface definitions for core data structures
 * - Type definitions for application-specific objects
 * - Configuration constants for YOLO, embedding, rendering, and tracking
 *
 * Centralizing types here allows for easier type checking and ensures
 * consistency across the codebase.
 */

// Type definitions for MTG Vision application

// Step interfaces for embedding processing
export interface Step {
  type: string;
  in_dim: number;
  out_dim: number;
  in_dtype: "float32" | "uint8";
  out_dtype: "float32" | "uint8";
}

export interface LinearTransform extends Step {
  type: "LinearTransform";
  params: {
    A: number[][];
    b: number[];
  };
}

export interface ScalarQuantizer extends Step {
  type: "ScalarQuantizer";
  params: {
    vmin: number[];
    vdiff: number[];
  };
  mode: "QT_8bit";
}

export interface Meta {
  model?: string;
  chain: LinearTransform[];
  quantize: ScalarQuantizer;
  ids: string[];
}

// Configuration Interfaces
export interface YoloConfig {
  modelUrl: string;
  inputWidth: number;
  inputHeight: number;
  confidenceThreshold: number;
  detBoxCoeffIndex: number; // Index where mask coefficients start
  protoMaskSize: number;
  maskCoeffCount: number;
}

export interface EmbeddingConfig {
  modelUrl: string;
  inputWidth: number;
  inputHeight: number;
  cropPaddingFactor: number;
}

export interface DrawingConfig {
  maskThreshold: number;
  maskCropPaddingFactor: number; // Padding used during mask generation affects drawing calculation
  maskDrawAlpha: number;
  colors: number[][];
}

export interface TrackerConfig {
  distanceThreshold: number;
  hitInertiaMin: number;
  hitInertiaMax: number;
  initDelay: number;
}

// Shared Interfaces
export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
} // Video coordinates

export interface BBoxCanvas {
  x: number;
  y: number;
  w: number;
  h: number;
} // Canvas coordinates

export interface VideoDims {
  w: number;
  h: number;
}

export interface RawDetection {
  bboxModel: number[]; // [x1, y1, x2, y2] in model input coords
  confidence: number;
  classId: number;
  maskCoeffs: any; // Shape [1, 1, num_coeffs], owned by caller of YoloProcessor.processFrame
}

export interface DrawInfo {
  id: number;
  color: number[];
  bboxCanvas: BBoxCanvas;
  label: string;
  mask?: {
    tensor: any; // Binary mask tensor
    videoBox: BBox; // Video coordinate box the mask corresponds to (reflecting padding used for generation)
  };
}

export interface ObjectEmbeddingInfo {
  id: number;
  hasBeenEmbedded: boolean;
  lastEmbeddingTime: number | null;
  creationTime: number;
  embedding: any | null; // Stores the embedding tensor (ownership with this object)
  lastKnownBboxVideo: BBox | null;
  matchId: string | null;
  lastMatchTime: number | null;
  cardData: {
    name: string;
    set_name: string;
    set_code: string;
    image_uris: {small: string};
  } | null;
}

// Constants (Configuration Objects)
export const YOLO_CONFIG: YoloConfig = {
  modelUrl: "/assets/models/yolov11s_seg__dk964hap__web_model/model.json",
  inputWidth: 640,
  inputHeight: 640,
  confidenceThreshold: 0.5,
  detBoxCoeffIndex: 6,
  protoMaskSize: 160,
  maskCoeffCount: 32,
};

export const EMBEDDING_CONFIG: EmbeddingConfig = {
  modelUrl:
    "/assets/models/convnextv2_convlinear__aivb8jvk-47500__encoder__web_model/model.json",
  inputWidth: 128,
  inputHeight: 192,
  cropPaddingFactor: 0.1,
};

export const DRAWING_CONFIG: DrawingConfig = {
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

export const TRACKER_CONFIG: TrackerConfig = {
  distanceThreshold: 100,
  hitInertiaMin: 0,
  hitInertiaMax: 15,
  initDelay: 2,
};

export const EMBEDDING_LOOP_INTERVAL_MS = 250;
