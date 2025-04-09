import * as SVG from 'svg.js';
import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import { $selectedDevice, $isStreaming, $detections, $selectedId, $devices, $status } from './util-store';
import { wsSendBlob, wsCanSend } from './util-websocket';
import {Detection, SvgInHtml} from "./types";

/**
 * Populates the $devices atom with available video input devices.
 * @async
 * @returns {Promise<void>}
 */
export async function populateDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    $devices.set(videoDevices);
  } catch (error) {
    console.error('Failed to populate devices:', error);
    $status.set('Error accessing devices.');
  }
}

/**
 * Represents a single card detection with its own SVG elements.
 */
class SvgCard {

  id: number;


  svg: SVG.Container;
  group: SVG.G;
  polygon: SVG.Polygon;
  textGroup: SVG.G;
  text: SVG.Text;

  onClick: (id: number) => void;

  /**
   * Creates a new Card instance.
   * @param {Object} detection - The detection data (e.g., { id, points, color, matches }).
   * @param {SVG.Container} svg - The SVG container to render the card in.
   * @param {Function} onClick - Callback function for handling clicks on the card.
   */
  constructor(detection: Detection, svg: SVG.Container, onClick: (id: number) => void) {
    this.id = detection.id;
    this.svg = svg;
    this.onClick = onClick;

    // Create SVG group and elements
    this.group = this.svg.group();
    this.polygon = this.group.polygon([])
      .fill('rgba(0, 255, 0, 0.0)')
      .stroke({ color: detection.color, width: 2 })
      .attr('pointer-events', 'all');
    this.textGroup = this.group.group();  // translate this
    this.text = this.textGroup.text('')  // rotate this
      .font({ size: 10, style: 'fill: white', family: 'goudy, serif' })

    // Attach click handler to the group
    this.group.on('click', (e) => {
      e.stopPropagation(); // Prevent bubbling to SVG background
      this.onClick(this.id);
    });
  }

  /**
   * Updates the card’s position and appearance.
   * @param {Object} detection - Updated detection data.
   * @param {boolean} isSelected - Whether the card is currently selected.
   */
  update(detection, isSelected) {
    // draw polygon
    const pointsStr = detection.points.map(p => p.join(',')).join(' ');
    this.polygon.plot(pointsStr);
    this.polygon.stroke({ color: isSelected ? 'yellow' : detection.color, width: isSelected ? 4 : 2 });

    // draw text
    const bestMatch = detection.matches[0];
    if (bestMatch) {
      const [[x0, y0], [x1, y1]] = detection.points.slice(0, 2);
      const angle = Math.atan2(y1 - y0, x1 - x0) * (180 / Math.PI);
      this.text.text(bestMatch.name)
      this.text.transform({rotation: angle, cx: 0, cy: 0});
      this.textGroup.transform({x: x0, y: y0})
    }
  }

  /**
   * Removes the card’s SVG elements from the container.
   */
  remove() {
    this.group.remove();
  }
}


class ComponentVideo extends LitElement {

  #selectedDeviceController = new StoreController(this, $selectedDevice);
  #isStreamingController = new StoreController(this, $isStreaming);
  #detectionsController = new StoreController(this, $detections);
  #selectedIdController = new StoreController(this, $selectedId);


  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
    }
    .container {
      position: relative;
      width: 100%;
      height: 100%;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      pointer-events: none;
    }
    svg {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: auto;
      z-index: 1;
    }
  `;

  currentStream: MediaStream | null;
  currentDeviceId: string | null;
  originalWidth: number | null;
  originalHeight: number | null;
  readyPromise: Promise<void>;
  resolveReady: () => void;
  cardMap: Map<number, SvgCard>;
  video: HTMLVideoElement;
  svgElement: SvgInHtml;
  svg: SVG.Container;
  sendInterval: number | null = null;

  constructor() {
    super();
    this.currentStream = null;
    this.currentDeviceId = null;
    this.originalWidth = null;
    this.originalHeight = null;
    this.readyPromise = new Promise(resolve => (this.resolveReady = resolve));
    this.cardMap = new Map(); // Store Card instances by detection ID
  }

  connectedCallback() {
    super.connectedCallback();
    this.tryAutoStart();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.currentStream) this.stopStream();
    window.removeEventListener('resize', this.updateOverlaySize);
  }

  render() {
    return html`
      <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <svg id="overlay"></svg>
        <stats-overlay/>
      </div>
    `;
  }

  firstUpdated() {
    this.video = this.shadowRoot.getElementById('video') as HTMLVideoElement;
    this.svgElement = this.shadowRoot.getElementById('overlay') as SvgInHtml;
    this.svg = SVG(this.svgElement);

    this.video.addEventListener('loadedmetadata', () => {
      this.originalWidth = this.video.videoWidth;
      this.originalHeight = this.video.videoHeight;
      this.updateOverlaySize();
      this.video.play().catch(e => console.error('Failed to play video:', e));
    });
    window.addEventListener('resize', this.updateOverlaySize);

    // Add click handler to SVG background to clear selection
    this.svg.on('click', (e) => {
      if (e.target === this.svgElement) { // Check if click is on the SVG itself, not a child
        $selectedId.set(null);
      }
    });

    this.resolveReady();
  }

  async updated() {
    await this.updateStream();
    this.drawDetections();
  }

  async tryAutoStart() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      this.currentStream = stream;
      const deviceId = stream.getVideoTracks()[0].getSettings().deviceId;
      await populateDevices();
      $selectedDevice.set(deviceId);
      $isStreaming.set(true);
      await this.readyPromise;
      this.video.srcObject = this.currentStream;
    } catch (error) {
      console.error('Auto-start failed:', error);
      $isStreaming.set(false);
      $status.set('Camera access failed: ' + error.message);
    }
  }

  async updateStream() {
    if (this.#isStreamingController.value) {
      if (!this.currentStream || this.#selectedDeviceController.value !== this.currentDeviceId) {
        await this.startStream(this.#selectedDeviceController.value);
      }
    } else if (this.currentStream) {
      this.stopStream();
    }
  }

  async startStream(deviceId) {
    if (this.currentStream) this.stopStream();
    try {
      const constraints = { video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: 640, height: 480 } };
      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      await this.readyPromise;
      this.video.srcObject = this.currentStream;
      this.currentDeviceId = deviceId;
      this.startSendingFrames();
    } catch (error) {
      console.error('Failed to start stream:', error);
      $status.set('Failed to start camera: ' + error.message);
    }
  }

  stopStream() {
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop());
      this.currentStream = null;
    }
    this.video.srcObject = null;
    if (this.sendInterval) {
      clearInterval(this.sendInterval);
      this.sendInterval = null;
    }
  }

  startSendingFrames() {
    if (this.sendInterval) return;
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    this.sendInterval = setInterval(() => {
      if (wsCanSend()) {
        ctx.drawImage(this.video, 0, 0, 640, 480);
        canvas.toBlob(wsSendBlob, 'image/jpeg', 0.5);
      }
    }, 100);
  }

  drawDetections() {
    if (!this.originalWidth || !this.originalHeight) return;

    const currentIds = new Set();

    this.#detectionsController.value.forEach(det => {
      const id = det.id;
      currentIds.add(id);
      let card = this.cardMap.get(id);

      if (!card) {
        card = new SvgCard(det, this.svg, (clickedId) => {
          // Only select if not already selected
          if (this.#selectedIdController.value !== clickedId) {
            $selectedId.set(clickedId);
          }
          // Do nothing if already selected, keeping it clicked
        });
        this.cardMap.set(id, card);
      }

      const isSelected = id === this.#selectedIdController.value;
      card.update(det, isSelected);
    });

    this.cardMap.forEach((card, id) => {
      if (!currentIds.has(id)) {
        card.remove();
        this.cardMap.delete(id);
      }
    });
  }

  updateOverlaySize = () => {
    if (this.originalWidth && this.originalHeight) {
      this.svg.viewbox(0, 0, this.originalWidth, this.originalHeight);
    }
  };
}

customElements.define('video-container', ComponentVideo);
