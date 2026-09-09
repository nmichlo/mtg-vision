/**
 * Embedding Service
 *
 * Provides functionality for card embedding and identification.
 * This file contains:
 * - EmbeddingProcessor: Creates embeddings from video frame regions
 * - Card data fetching from Scryfall API
 *
 * The embedding process extracts visual features from cropped card images
 * which are then used to match against a database of known card embeddings
 * for identification.
 */

import * as tf from "@tensorflow/tfjs";
import { BBox, EmbeddingConfig } from "./types";

/**
 * Processes card embeddings for identification
 */
export class EmbeddingProcessor {
  constructor(
    private embedModel: tf.GraphModel,
    private config: EmbeddingConfig,
  ) {}

  /**
   * Creates an embedding tensor from a video frame region
   */
  async createEmbedding(
    videoElement: HTMLVideoElement,
    targetVideoBox: BBox,
  ): Promise<tf.Tensor | null> {
    if (!videoElement.videoWidth || !this.embedModel) return null;
    let embedding: tf.Tensor | null = null;
    let clonedEmbedding: tf.Tensor | null = null;
    try {
      const vW = videoElement.videoWidth;
      const vH = videoElement.videoHeight;
      const bW = targetVideoBox.x2 - targetVideoBox.x1;
      const bH = targetVideoBox.y2 - targetVideoBox.y1;
      const pX = bW * this.config.cropPaddingFactor;
      const pY = bH * this.config.cropPaddingFactor;
      const cX1 = Math.max(0, Math.floor(targetVideoBox.x1 - pX));
      const cY1 = Math.max(0, Math.floor(targetVideoBox.y1 - pY));
      const cX2 = Math.min(vW, Math.ceil(targetVideoBox.x2 + pX));
      const cY2 = Math.min(vH, Math.ceil(targetVideoBox.y2 + pY));
      if (cX2 <= cX1 || cY2 <= cY1) {
        console.warn("Invalid crop embed.");
        return null;
      }
      const boxes = [[cY1 / vH, cX1 / vW, cY2 / vH, cX2 / vW]];
      const cropSize: [number, number] = [
        this.config.inputHeight,
        this.config.inputWidth,
      ];
      embedding = await tf.tidy(() => {
        const f = tf.browser.fromPixels(videoElement);
        const c = tf.image.cropAndResize(
          f.expandDims(0).toFloat() as tf.Tensor4D,
          boxes,
          [0],
          cropSize,
          "bilinear",
        );
        return this.embedModel.execute(c.div(255.0)) as tf.Tensor;
      });
      if (embedding?.shape?.length > 0) {
        clonedEmbedding = embedding.clone();
      } else {
        console.warn("Embed invalid tensor.");
      }
    } catch (error) {
      console.error("Error embedding:", error);
    } finally {
      embedding?.dispose();
    }
    return clonedEmbedding;
  }
}

/**
 * Fetches card data from Scryfall API
 */
export async function fetchCardDataFromScryfall(cardId: string) {
  try {
    const response = await fetch(`https://api.scryfall.com/cards/${cardId}`, {
      headers: {
        'Cache-Control': 'max-age=86400', // Cache for 24 hours
        'Accept': 'application/json',
      },
    });
    if (!response.ok) {
      console.warn(`Failed to fetch card data for ${cardId}: ${response.statusText}`);
      return null;
    }
    const cardData = await response.json();
    return cardData;
  } catch (error) {
    console.error(`Error fetching card data for ${cardId}:`, error);
    return null;
  }
}
