/**
 * Canvas Rendering Service
 *
 * Provides canvas rendering functionality for the MTG Vision application.
 * This file includes:
 * - CanvasRenderer: Manages drawing on canvas overlays
 * - Utilities for scaling between video and canvas coordinates
 * - Mask rendering functionality for object segmentation
 *
 * The rendering service handles visualization of detection results,
 * including bounding boxes, labels, and segmentation masks on top
 * of the video feed.
 */

import * as tf from "@tensorflow/tfjs";
import { BBox, BBoxCanvas, DrawInfo, DrawingConfig } from "./types";

/**
 * Handles canvas rendering for video overlay
 */
export class CanvasRenderer {
  private visibleCtx: CanvasRenderingContext2D;
  private bufferCanvas: HTMLCanvasElement;
  private bufferCtx: CanvasRenderingContext2D;
  private displayWidth = 0;
  private displayHeight = 0;
  private videoWidth = 0;
  private videoHeight = 0;
  public scale = 1;
  public offsetX = 0;
  public offsetY = 0;

  constructor(
    private visibleCanvas: HTMLCanvasElement,
    private drawingConfig: DrawingConfig,
  ) {
    const ctx = visibleCanvas.getContext("2d");
    if (!ctx) throw new Error("No visible ctx");
    this.visibleCtx = ctx;
    this.bufferCanvas = document.createElement("canvas");
    const bufCtx = this.bufferCanvas.getContext("2d");
    if (!bufCtx) throw new Error("No buffer ctx");
    this.bufferCtx = bufCtx;
  }

  /**
   * Updates canvas dimensions based on video element
   */
  updateDimensions(videoElement: HTMLVideoElement): boolean {
    if (!videoElement.videoWidth) return false;
    const dW = videoElement.clientWidth;
    const dH = videoElement.clientHeight;
    const vW = videoElement.videoWidth;
    const vH = videoElement.videoHeight;
    let changed = false;
    if (this.displayWidth !== dW || this.displayHeight !== dH) {
      this.visibleCanvas.width = dW;
      this.visibleCanvas.height = dH;
      this.bufferCanvas.width = dW;
      this.bufferCanvas.height = dH;
      this.displayWidth = dW;
      this.displayHeight = dH;
      changed = true;
    }
    if (this.videoWidth !== vW || this.videoHeight !== vH || changed) {
      this.videoWidth = vW;
      this.videoHeight = vH;
      const sX = dW / vW;
      const sY = dH / vH;
      this.scale = Math.min(sX, sY);
      this.offsetX = (dW - vW * this.scale) / 2;
      this.offsetY = (dH - vH * this.scale) / 2;
      changed = true;
    }
    return changed;
  }

  /**
   * Converts video coordinates to canvas coordinates
   */
  convertVideoBoxToCanvas(videoBox: BBox): BBoxCanvas {
    return {
      x: videoBox.x1 * this.scale + this.offsetX,
      y: videoBox.y1 * this.scale + this.offsetY,
      w: (videoBox.x2 - videoBox.x1) * this.scale,
      h: (videoBox.y2 - videoBox.y1) * this.scale,
    };
  }

  /**
   * Clears all canvas drawing
   */
  clearAll(): void {
    this.visibleCtx.clearRect(
      0,
      0,
      this.displayWidth,
      this.displayHeight,
    );
    this.bufferCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
  }

  /**
   * Draws a frame with all detected objects
   */
  async drawFrame(objectsToDraw: DrawInfo[]): Promise<void> {
    this.bufferCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
    const maskPromises: Promise<void>[] = [];
    for (const obj of objectsToDraw) {
      this.drawBoundingBox(
        this.bufferCtx,
        obj.bboxCanvas,
        obj.color,
        obj.label,
      );
      if (obj.mask?.tensor && !obj.mask.tensor.isDisposed) {
        maskPromises.push(
          this.drawMask(
            this.bufferCtx,
            obj.mask.tensor,
            obj.color,
            obj.mask.videoBox,
          ),
        );
      }
    }
    await Promise.all(maskPromises);
    this.visibleCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
    this.visibleCtx.drawImage(this.bufferCanvas, 0, 0);
  }

  /**
   * Draws a bounding box with label
   */
  private drawBoundingBox(
    ctx: CanvasRenderingContext2D,
    bbox: BBoxCanvas,
    color: number[],
    label: string,
  ): void {
    ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox.x, bbox.y, bbox.w, bbox.h);
    ctx.font = "12px sans-serif";
    ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.8)`;
    const m = ctx.measureText(label);
    const tW = m.width;
    const tH = 12;
    const p = 2;
    let tY = bbox.y - p - 1;
    let bY = tY - tH - p;
    if (bY < 0) {
      tY = bbox.y + bbox.h + tH + p;
      bY = bbox.y + bbox.h + p;
    }
    ctx.fillRect(bbox.x - 1, bY, tW + p * 2, tH + p * 2);
    ctx.fillStyle = `white`;
    ctx.fillText(label, bbox.x + p - 1, tY);
  }

  /**
   * Draws a mask for an object
   */
  private async drawMask(
    drawCtx: CanvasRenderingContext2D,
    maskTensor: tf.Tensor2D,
    color: number[],
    videoBox: BBox,
  ): Promise<void> {
    let tempMaskCanvas: HTMLCanvasElement | null = null;
    const tensorsToDispose: tf.Tensor[] = [];
    try {
      const [mH, mW] = maskTensor.shape;
      const bW = videoBox.x2 - videoBox.x1;
      const bH = videoBox.y2 - videoBox.y1;
      const pX = bW * this.drawingConfig.maskCropPaddingFactor;
      const pY = bH * this.drawingConfig.maskCropPaddingFactor;
      const mX1 = Math.max(0, videoBox.x1 - pX);
      const mY1 = Math.max(0, videoBox.y1 - pY);
      const vW = bW + 2 * pX;
      const vH = bH + 2 * pY; // Effective video size of mask area
      const colorTensor = tf.tensor1d(color.map((c) => c / 255.0));
      tensorsToDispose.push(colorTensor);
      const maskColored = tf.tidy(() =>
        maskTensor.expandDims(-1).mul(colorTensor.reshape([1, 1, 3])),
      );
      tensorsToDispose.push(maskColored);
      const alphaChannel = maskTensor.expandDims(-1);
      tensorsToDispose.push(alphaChannel);
      const maskRgba = tf.tidy(() =>
        tf.concat([maskColored, alphaChannel], -1),
      );
      tensorsToDispose.push(maskRgba);
      const maskPixelData = await tf.browser.toPixels(maskRgba as tf.Tensor2D | tf.Tensor3D);
      const imageData = new ImageData(maskPixelData, mW, mH);
      tempMaskCanvas = document.createElement("canvas");
      tempMaskCanvas.width = mW;
      tempMaskCanvas.height = mH;
      const tempCtx = tempMaskCanvas.getContext("2d");
      if (!tempCtx) throw new Error("No temp ctx");
      tempCtx.putImageData(imageData, 0, 0);
      const cX = mX1 * this.scale + this.offsetX;
      const cY = mY1 * this.scale + this.offsetY;
      const cW = vW * this.scale;
      const cH = vH * this.scale;
      drawCtx.save();
      drawCtx.globalAlpha = this.drawingConfig.maskDrawAlpha;
      if (cW > 0 && cH > 0) {
        drawCtx.drawImage(tempMaskCanvas, cX, cY, cW, cH);
      }
      drawCtx.restore();
    } catch (error) {
      console.error("Error drawing mask:", error);
    } finally {
      tf.dispose(tensorsToDispose);
    }
  }
}
