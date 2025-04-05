import SVG from 'https://esm.run/svg.js';
import { LitElement, html, css } from 'https://esm.run/lit';
import { StoreController } from 'https://esm.run/@nanostores/lit';
import { $selectedDevice, $isStreaming, $detections, $selectedId, $devices, $status } from './store.js';
import { ws } from './websocket.js';

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

class VideoContainer extends LitElement {
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
    }
    svg {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }
  `;

  #selectedDeviceController = new StoreController(this, $selectedDevice);
  #isStreamingController = new StoreController(this, $isStreaming);
  #detectionsController = new StoreController(this, $detections);
  #selectedIdController = new StoreController(this, $selectedId);

  constructor() {
    super();
    this.currentStream = null;
    this.currentDeviceId = null;
    this.originalWidth = null;
    this.originalHeight = null;
    this.readyPromise = new Promise(resolve => (this.resolveReady = resolve));
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
      </div>
    `;
  }

  firstUpdated() {
    this.video = this.shadowRoot.getElementById('video');
    this.svgElement = this.shadowRoot.getElementById('overlay');
    this.svg = SVG(this.svgElement);
    this.video.addEventListener('loadedmetadata', () => {
      console.log('Metadata loaded');
      this.originalWidth = this.video.videoWidth;
      this.originalHeight = this.video.videoHeight;
      this.updateOverlaySize();
      this.video.play().then(() => {
        console.log('Video playing');
      }).catch(e => {
        console.error('Failed to play video in loadedmetadata:', e);
      });
    });
    window.addEventListener('resize', this.updateOverlaySize);
    this.resolveReady();
  }

  async updated() {
    console.log('updated() called, isStreaming:', this.#isStreamingController.value);
    await this.updateStream();
    this.drawDetections();
  }

  async tryAutoStart() {
    try {
      console.log('Attempting auto-start');
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      this.currentStream = stream;
      const deviceId = stream.getVideoTracks()[0].getSettings().deviceId;
      await populateDevices();
      $selectedDevice.set(deviceId);
      $isStreaming.set(true);
      await this.readyPromise;
      this.video.srcObject = this.currentStream;
      console.log('Stream set in tryAutoStart');
    } catch (error) {
      console.error('Auto-start failed:', error);
      $isStreaming.set(false);
      $status.set('Camera access failed: ' + error.message);
    }
  }

  async updateStream() {
    console.log('updateStream() called, isStreaming:', this.#isStreamingController.value);
    if (this.#isStreamingController.value) {
      if (!this.currentStream || this.#selectedDeviceController.value !== this.currentDeviceId) {
        await this.startStream(this.#selectedDeviceController.value);
      }
    } else if (this.currentStream) {
      this.stopStream();
    }
  }

  async startStream(deviceId) {
    console.log('startStream() called with deviceId:', deviceId);
    if (this.currentStream) {
      this.stopStream();
    }
    try {
      const constraints = { video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: 640, height: 480 } };
      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('Stream obtained:', this.currentStream);
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
    console.log('stopStream() called');
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
      if (ws && ws.readyState === WebSocket.OPEN) {
        ctx.drawImage(this.video, 0, 0, 640, 480);
        canvas.toBlob(blob => ws.send(blob), 'image/jpeg', 0.5);
      }
    }, 100);
  }

  drawDetections() {
    this.svg.clear();
    if (!this.originalWidth || !this.originalHeight) return;
    const scaleX = this.originalWidth / 640;
    const scaleY = this.originalHeight / 480;
    this.#detectionsController.value.forEach(det => {
      const scaledPoints = det.points.map(p => [p[0] * scaleX, p[1] * scaleY]);
      const pointsStr = scaledPoints.map(p => p.join(',')).join(' ');
      const isSelected = det.id === this.#selectedIdController.value;
      this.svg.polygon(pointsStr)
        .fill('transparent') // Make interior clickable
        .stroke({ color: isSelected ? 'yellow' : det.color, width: isSelected ? 4 : 2 })
        .attr('pointer-events', 'all') // Allow clicks on interior
        .on('click', () => {
          const currentSelectedId = this.#selectedIdController.value;
          if (currentSelectedId === det.id) {
            $selectedId.set(null);
          } else {
            $selectedId.set(det.id);
          }
        });
      const bestMatch = det.matches[0];
      if (bestMatch) {
        const topPoint = scaledPoints.reduce((a, b) => a[1] < b[1] ? a : b);
        this.svg.text(bestMatch.name)
          .move(topPoint[0], topPoint[1] - 15)
          .font({ fill: 'white', size: 12 });
      }
    });
  }

  updateOverlaySize = () => {
    if (this.originalWidth && this.originalHeight) {
      this.svg.viewbox(0, 0, this.originalWidth, this.originalHeight);
    }
  };
}

customElements.define('video-container', VideoContainer);
