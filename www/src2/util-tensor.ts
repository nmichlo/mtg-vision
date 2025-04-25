/**
 * TensorFlow Utility Functions
 *
 * Provides low-level tensor manipulation utilities for the MTG Vision application.
 * This file includes:
 * - Vector similarity functions (cosine similarity)
 * - Linear transformation functions for embeddings
 * - Quantization functions for efficient storage and comparison
 * - Mask generation utilities for segmentation visualization
 *
 * These utilities support the core functionality of object detection,
 * segmentation, and embedding generation throughout the application.
 */

import * as tf from "@tensorflow/tfjs";
import {
  BBox,
  VideoDims,
  LinearTransform,
  ScalarQuantizer
} from "./types";

// Vector similarity functions
export const cosineSimilarity = (a: number[], b: number[]) => {
  let A = 0;
  let B = 0;
  let AB = 0;
  for (let i = 0; i < a.length; i++) {
    A += a[i] * a[i];
    B += b[i] * b[i];
    AB += a[i] * b[i];
  }
  if (A === 0 || B === 0) {
    return 1;
  }
  return AB / (Math.sqrt(A) * Math.sqrt(B));
};

export const cosineSimilarityPreNorm = (a: number[], b: number[]) => {
  let AB = 0;
  for (let i = 0; i < a.length; i++) {
    AB += a[i] * b[i];
  }
  return AB;
};

// Linear transformation functions
export const makeLinearForward = (params: LinearTransform) => {
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const b = params.params.b;
  const A = params.params.A;

  return (vec: number[]): number[] => {
    // compute
    const output = new Array(outDim);
    for (let i = 0; i < outDim; i++) {
      let total = b[i];
      for (let j = 0; j < inDim; j++) {
        total += A[i][j] * vec[j];
      }
      output[i] = total;
    }
    return output;
  };
};

// Quantizer functions
export const makeEncode = (params: ScalarQuantizer) => {
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const vmin = params.params.vmin;
  const vdiff = params.params.vdiff;

  if (inDim !== outDim) {
    throw new Error("in_dim != out_dim");
  }

  return (vec: number[]): Uint8Array => {
    // compute
    const out = new Uint8Array(outDim);
    for (let i = 0; i < inDim; i++) {
      const vd = vdiff[i];
      const vm = vmin[i];
      let xi = 0;
      if (vd !== 0) {
        xi = (vec[i] - vm) / vd;
        if (xi < 0) {
          xi = 0;
        }
        if (xi > 1.0) {
          xi = 1.0;
        }
      }
      out[i] = Math.round(xi * 255);
    }
    return out;
  };
};

export const makeDecode = (params: ScalarQuantizer) => {
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const vmin = params.params.vmin;
  const vdiff = params.params.vdiff;

  if (inDim !== outDim) {
    throw new Error("in_dim != out_dim");
  }

  return (vec: Uint8Array): number[] => {
    const out = new Array(outDim);
    for (let i = 0; i < inDim; i++) {
      out[i] = (vmin[i] + (vec[i] + 0.5) / 255) * vdiff[i];
    }
    return out;
  };
};

export const makeDecodeSimilarityPreNorm = (params: ScalarQuantizer) => {
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const vmin = params.params.vmin;
  const vdiff = params.params.vdiff;

  if (inDim !== outDim) {
    throw new Error("in_dim != out_dim");
  }

  return (a: Uint8Array, b: Uint8Array) => {
    let AB = 0;
    for (let i = 0; i < inDim; i++) {
      const A = (vmin[i] + (a[i] + 0.5) / 255) * vdiff[i];
      const B = (vmin[i] + (b[i] + 0.5) / 255) * vdiff[i];
      AB += A * B;
    }
    return AB;
  };
};

export const makeDecodeSimilarity = (params: ScalarQuantizer) => {
  const inDim = params.in_dim;
  const outDim = params.out_dim;
  const vmin = params.params.vmin;
  const vdiff = params.params.vdiff;

  if (inDim !== outDim) {
    throw new Error("in_dim != out_dim");
  }

  return (a: Uint8Array, b: Uint8Array) => {
    let AB = 0;
    let AA = 0;
    let BB = 0;
    for (let i = 0; i < inDim; i++) {
      const A = (vmin[i] + (a[i] + 0.5) / 255) * vdiff[i];
      const B = (vmin[i] + (b[i] + 0.5) / 255) * vdiff[i];
      AB += A * B;
      AA += A * A;
      BB += B * B;
    }
    return AB / (Math.sqrt(AA) * Math.sqrt(BB));
  };
};

/**
 * Generates a binary mask tensor for a single detection using a functional approach.
 * Disposes intermediate tensors it creates via tf.tidy.
 * Does NOT dispose input tensors (protoTensor, maskCoeffs).
 * Returns null if generation fails or inputs are invalid.
 */
export async function generateMask(
  protoTensor: tf.Tensor,
  maskCoeffs: tf.Tensor,
  targetVideoBox: BBox, // The UNPADDED video box of the detection
  videoDims: VideoDims,
  config: {
    protoSize: number;
    coeffCount: number;
    paddingFactor: number;
    threshold: number;
  },
): Promise<tf.Tensor2D | null> {
  // Input validation
  if (
    !protoTensor ||
    protoTensor.isDisposed ||
    !maskCoeffs ||
    maskCoeffs.isDisposed
  ) {
    console.warn("generateMask received disposed tensor.");
    return null;
  }
  if (!targetVideoBox || !videoDims || videoDims.w <= 0 || videoDims.h <= 0) {
    console.warn("generateMask received invalid bbox or video dimensions.");
    return null;
  }

  let finalMask: tf.Tensor2D | null = null;
  try {
    finalMask = tf.tidy(() => {
      // 1. Combine protos and coefficients
      const protosReshaped = protoTensor
        .squeeze([0])
        .reshape([config.protoSize * config.protoSize, config.coeffCount]);
      const coeffsReshaped = maskCoeffs.reshape([config.coeffCount, 1]);
      const maskProto = tf.matMul(protosReshaped, coeffsReshaped);
      const maskReshaped = maskProto.reshape([
        config.protoSize,
        config.protoSize,
      ]);
      const maskActivated = tf.sigmoid(maskReshaped);

      // 2. Calculate Crop Box based on TARGET video box and padding
      const boxWidth = targetVideoBox.x2 - targetVideoBox.x1;
      const boxHeight = targetVideoBox.y2 - targetVideoBox.y1;
      const padX = boxWidth * config.paddingFactor;
      const padY = boxHeight * config.paddingFactor;
      const cropX1 = Math.max(0, targetVideoBox.x1 - padX);
      const cropY1 = Math.max(0, targetVideoBox.y1 - padY);
      const cropX2 = Math.min(videoDims.w, targetVideoBox.x2 + padX);
      const cropY2 = Math.min(videoDims.h, targetVideoBox.y2 + padY);
      // Target dimensions for the output mask tensor
      const targetCropW = Math.ceil(cropX2 - cropX1);
      const targetCropH = Math.ceil(cropY2 - cropY1);

      if (targetCropW <= 0 || targetCropH <= 0) return null;

      // Normalized coordinates of the area to crop *from the prototype mask*
      const normalizedPaddedBbox = [
        [
          cropY1 / videoDims.h,
          cropX1 / videoDims.w,
          cropY2 / videoDims.h,
          cropX2 / videoDims.w,
        ],
      ];

      // 3. Crop and Resize the activated prototype mask
      const maskExpanded = maskActivated.expandDims(0).expandDims(-1) as tf.Tensor4D;
      const maskCroppedResized = tf.image.cropAndResize(
        maskExpanded,
        normalizedPaddedBbox,
        [0],
        [targetCropH, targetCropW],
        "bilinear",
      );

      // 4. Threshold to get binary mask
      const mask = maskCroppedResized
        .squeeze([0, 3])
        .greater(config.threshold)
        .cast("float32") as tf.Tensor2D;
      return mask; // Return the final tensor from tidy
    }) as tf.Tensor2D;
  } catch (error) {
    console.error("Error generating mask:", error);
    finalMask?.dispose();
    return null;
  }
  // Ownership of finalMask (if not null) is transferred to the caller.
  return finalMask;
}
