/**
 * Machine Learning Model Service
 *
 * Provides functionality for loading and using TensorFlow.js models in the browser.
 * This file includes:
 * - ModelLoader: Static utility for loading ML models
 * - YoloProcessor: Processes video frames with YOLO segmentation model
 *
 * The service handles model loading, preprocessing of inputs, and postprocessing
 * of model outputs for the object detection and segmentation pipeline.
 */

import * as tf from "@tensorflow/tfjs";
import { RawDetection, YoloConfig } from "./types";

/**
 * Utility for loading TensorFlow.js models
 */
export class ModelLoader {
  static async loadModel(url: string): Promise<tf.GraphModel> {
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

/**
 * Processes video frames with a YOLO object detection model
 */
export class YoloProcessor {
  constructor(
    private yoloModel: tf.GraphModel,
    private config: YoloConfig,
  ) {}

  /**
   * Processes a video frame to detect objects
   * Returns detection results and prototype mask tensor for mask generation
   */
  async processFrame(
    frameTensor: tf.Tensor,
  ): Promise<{ detections: RawDetection[]; protoTensor: tf.Tensor | null }> {
    let output: tf.Tensor[] | null = null;
    const detections: RawDetection[] = [];
    let protoTensor: tf.Tensor | null = null;
    const tensorsCreated: tf.Tensor[] = [];
    try {
      const inputTensor = tf.tidy(() => {
        const r = tf.image.resizeBilinear(frameTensor as tf.Tensor3D, [
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
