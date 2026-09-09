/**
 * Video Service
 *
 * Provides functionality for managing video camera streams in the browser.
 * This service handles:
 * - Starting/stopping camera streams
 * - Providing access to the video element
 * - Managing stream errors and readiness state
 * - Retrieving video dimensions
 *
 * Acts as a wrapper around the browser's MediaDevices API to simplify
 * camera access for the rest of the application.
 */

import { VideoDims } from "./types";

/**
 * Manages video streams from camera or other sources
 */
export class VideoSourceManager {
  private videoElement: HTMLVideoElement | null = null;
  private currentStream: MediaStream | null = null;
  private streamError: string | null = null;

  constructor() {}

  /**
   * Starts a video stream from the camera
   */
  async startStream(
    element: HTMLVideoElement,
    constraints?: MediaStreamConstraints,
  ): Promise<void> {
    this.videoElement = element;
    this.streamError = null;
    if (this.currentStream) {
      this.stopStream();
    }
    try {
      const defaultConstraints = {
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      };
      this.currentStream = await navigator.mediaDevices.getUserMedia(
        constraints || defaultConstraints,
      );
      if (this.videoElement) {
        this.videoElement.srcObject = this.currentStream;
        await this.videoElement.play();
        console.log("Video stream started.");
      } else {
        throw new Error("Video element not set.");
      }
    } catch (error: any) {
      console.error("Camera failed:", error);
      this.streamError = `Camera failed: ${error.message}`;
      this.stopStream();
      throw error;
    }
  }

  /**
   * Stops the current video stream
   */
  stopStream(): void {
    if (this.currentStream) {
      this.currentStream.getTracks().forEach((track) => track.stop());
    }
    this.currentStream = null;
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.srcObject = null;
    }
  }

  /**
   * Gets the current video element
   */
  getVideoElement(): HTMLVideoElement | null {
    return this.videoElement;
  }

  /**
   * Gets the current video dimensions
   */
  getVideoDimensions(): VideoDims | null {
    if (this.videoElement?.videoWidth) {
      return {
        w: this.videoElement.videoWidth,
        h: this.videoElement.videoHeight,
      };
    }
    return null;
  }

  /**
   * Checks if the video stream is ready
   */
  isReady(): boolean {
    return !!(
      this.videoElement &&
      this.currentStream &&
      this.videoElement.readyState >= 4 &&
      !this.videoElement.paused
    );
  } // Use readyState 4 (HAVE_ENOUGH_DATA)

  /**
   * Gets the current stream error if any
   */
  getError(): string | null {
    return this.streamError;
  }
}
