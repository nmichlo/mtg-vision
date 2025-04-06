import SVG from 'https://esm.run/svg.js';
import { LitElement, html, css } from 'https://esm.run/lit';
import { StoreController } from 'https://esm.run/@nanostores/lit';
import { $selectedDevice, $isStreaming, $detections, $selectedId, $devices, $status } from './util-store.js';
import { wsSendBlob, wsCanSend } from './util-websocket.js';

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
  /**
   * Creates a new Card instance.
   * @param {Object} detection - The detection data (e.g., { id, points, color, matches }).
   * @param {SVG.Container} svg - The SVG container to render the card in.
   * @param {Function} onClick - Callback function for handling clicks on the card.
   */
  constructor(detection, svg, onClick) {
    this.id = detection.id;
    this.svg = svg;
    this.onClick = onClick;

    // Create SVG group and elements
    this.group = this.svg.group();
    this.polygon = this.group.polygon()
      .fill('rgba(0, 255, 0, 0.2)')
      .stroke({ color: detection.color, width: 2 })
      .attr('pointer-events', 'all');
    this.text = this.group.text('')
      .font({ fill: 'white', size: 12 });

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
   * @param {number} scaleX - Scaling factor for X coordinates.
   * @param {number} scaleY - Scaling factor for Y coordinates.
   */
  update(detection, isSelected, scaleX, scaleY) {
    // Update polygon points
    const scaledPoints = detection.points.map(p => [p[0] * scaleX, p[1] * scaleY]);
    const pointsStr = scaledPoints.map(p => p.join(',')).join(' ');
    this.polygon.plot(pointsStr);
    this.polygon.stroke({ color: isSelected ? 'yellow' : detection.color, width: isSelected ? 4 : 2 });

    // Update text (e.g., best match name)
    const bestMatch = detection.matches[0];
    if (bestMatch) {
      const [p1, p2] = scaledPoints.slice(0, 2);
      const midX = (p1[0] + p2[0]) / 2;
      const midY = (p1[1] + p2[1]) / 2;
      const angle = Math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * (180 / Math.PI);
      this.text.text(bestMatch.name).move(midX, midY).rotate(angle, midX, midY);
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
    this.video = this.shadowRoot.getElementById('video');
    this.svgElement = this.shadowRoot.getElementById('overlay');
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

    const scaleX = this.originalWidth / 640;
    const scaleY = this.originalHeight / 480;
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
      card.update(det, isSelected, scaleX, scaleY);
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
