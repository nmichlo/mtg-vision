import SVG from 'https://esm.run/svg.js';
import { LitElement, html, css } from 'https://esm.run/lit';
import { StoreController } from 'https://esm.run/@nanostores/lit';
import { $selectedDevice, $isStreaming, $detections, $selectedId, $devices, $status } from './store.js';

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
  static styles = css`/* ... */`;

  #selectedDeviceController = new StoreController(this, $selectedDevice);
  #isStreamingController = new StoreController(this, $isStreaming);
  #detectionsController = new StoreController(this, $detections);
  #selectedIdController = new StoreController(this, $selectedId);

  constructor() {
    super();
    this.currentStream = null;
    this.currentDeviceId = null;
  }

  // connectedCallback() {
  //   super.connectedCallback();
  //   this.tryAutoStart();
  // }

  // disconnectedCallback() {
  //   super.disconnectedCallback();
  //   if (this.currentStream) this.stopStream();
  //   window.removeEventListener('resize', this.updateOverlaySize);
  // }

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
    this.updateOverlaySize();
    window.addEventListener('resize', this.updateOverlaySize);
    this.video.addEventListener('loadedmetadata', () => {
      this.video.play();
    });
  }

  async tryAutoStart() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      this.video.srcObject = stream;
      this.currentStream = stream;
      const deviceId = stream.getVideoTracks()[0].getSettings().deviceId;
      await populateDevices();
      $selectedDevice.set(deviceId);
      $isStreaming.set(true);
    } catch (error) {
      console.error('Auto-start failed:', error);
      $isStreaming.set(false);
      $status.set('Camera access failed.');
    }
  }

  async updated() {
    await this.updateStream();
    this.drawDetections();
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
    if (this.currentStream) {
      this.stopStream();
    }
    try {
      const constraints = {video: {deviceId: {exact: deviceId}, width: 640, height: 480}};
      if (!deviceId) {
        delete constraints.video.deviceId;
      }
      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.currentStream;
      this.currentDeviceId = deviceId;
      this.startSendingFrames();
    } catch (error) {
      console.error('Failed to start stream:', error);
      $status.set('Failed to start camera.' + error);
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
      if (ws && ws.readyState === WebSocket.OPEN) {
        ctx.drawImage(this.video, 0, 0, 640, 480);
        canvas.toBlob(blob => ws.send(blob), 'image/jpeg', 0.5);
      }
    }, 100);
  }

  drawDetections() {
    this.svg.clear();
    this.#detectionsController.value.forEach(det => {
      const isSelected = det.id === this.#selectedIdController.value;
      const points = det.points.map(p => p.join(',')).join(' ');
      this.svg.polygon(points)
        .fill('none')
        .stroke({ color: isSelected ? 'yellow' : det.color, width: isSelected ? 4 : 2 })
        .attr('pointer-events', 'auto')
        .on('click', () => $selectedId.set(det.id));
      const bestMatch = det.matches[0];
      if (bestMatch) {
        const topPoint = det.points.reduce((a, b) => a[1] < b[1] ? a : b);
        this.svg.text(bestMatch.name)
          .move(topPoint[0], topPoint[1] - 15)
          .font({ fill: 'white', size: 12 });
      }
    });
  }

  updateOverlaySize = () => {
    const { width, height } = this.video.getBoundingClientRect();
    this.svg.viewbox(0, 0, width, height);
  };
}

customElements.define('video-container', VideoContainer);
