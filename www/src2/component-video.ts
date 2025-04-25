/**
 * Video Container Component
 *
 * Main component that creates the user interface for the MTG Vision application.
 * Provides the container for displaying:
 * - The camera video feed
 * - An overlay canvas for drawing bounding boxes and segmentation masks
 * - Status information
 * - Sidebar with card previews
 *
 * This component delegates all vision processing logic to the VisionController
 * and focuses purely on rendering the UI and handling user interactions.
 */

import { css, html, LitElement } from "lit";
import { customElement, query, state } from "lit/decorators.js";
import "./component-card-preview";
import { VisionController } from "./vision-controller";
import { TrackedObjectState } from "./service-tracker";

/**
 * Main video component that orchestrates detection, tracking, and identification
 */
@customElement("video-container")
export class VideoContainer extends LitElement {
  static styles = css`
    :host {
      display: flex;
      width: 100%;
      height: 100%;
      overflow: hidden;
      position: relative;
      background-color: #222;
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
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background-color: rgba(0, 0, 0, 0.7);
      color: white;
      font-size: 0.9em;
      padding: 5px 10px;
      text-align: center;
      z-index: 10;
    }
    .sidebar {
      position: absolute;
      right: 0;
      top: 0;
      bottom: 0;
      width: 150px; /* Reduced width */
      background-color: rgba(0, 0, 0, 0.7);
      padding: 10px;
      overflow-y: auto;
      z-index: 10;
    }
  `;

  // --- Element Refs ---
  @query("#video") private videoElement!: HTMLVideoElement;
  @query("#overlay-canvas") private canvasElement!: HTMLCanvasElement;

  // --- State ---
  @state() private inferenceStatus: string = "Initializing...";
  @state() private cardPreviews: { id: number; image: string; timestamp: number }[] = [];

  // --- Controller ---
  private visionController!: VisionController;

  // --- Lit Lifecycle ---
  connectedCallback() {
    super.connectedCallback();
    this.visionController = new VisionController(
      this.updateStatus,
      this.updateCardPreview
    );
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.visionController.dispose();
    console.log("VideoContainer disconnected & cleaned up.");
  }

  protected firstUpdated() {
    this.initializeVision();
  }

  // --- Initialization ---
  private async initializeVision() {
    try {
      await this.visionController.initialize(
        this.videoElement,
        this.canvasElement
      );
    } catch (error) {
      console.error("Failed to initialize vision:", error);
      this.inferenceStatus = `Error: ${error instanceof Error ? error.message : error}`;
    }
  }

  // --- Callbacks ---
  private updateStatus = (status: string): void => {
    this.inferenceStatus = status;
  };

  private updateCardPreview = (id: number, image: string): void => {
    // Remove any existing preview for this ID
    this.cardPreviews = this.cardPreviews.filter(p => p.id !== id);

    // Add new preview
    this.cardPreviews = [
      ...this.cardPreviews,
      { id, image, timestamp: Date.now() }
    ];

    // Keep only the 5 most recent previews
    if (this.cardPreviews.length > 5) {
      this.cardPreviews = this.cardPreviews
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, 5);
    }
  };

  // --- Helper Methods ---
  private getTrackedObjectState(): TrackedObjectState {
    return this.visionController.getTrackedObjectState();
  }

  // --- Render ---
  render() {
    return html`
      <video id="video" muted playsinline></video>
      <canvas id="overlay-canvas"></canvas>
      <div class="status">${this.inferenceStatus}</div>
      <div class="sidebar">
        ${this.cardPreviews.map(preview => {
          const objectInfo = this.getTrackedObjectState().get(preview.id);
          return html`
            <card-preview
              objectId="${preview.id}"
              image="${preview.image}"
              .cardInfo="${objectInfo}"
            ></card-preview>
          `;
        })}
      </div>
    `;
  }
}
