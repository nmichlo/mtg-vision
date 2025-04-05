import SVG from 'https://esm.run/svg.js';
import { LitElement, html, css } from 'https://esm.run/lit';
import { $selectedDevice, $isStreaming, $detections, $selectedId, $devices } from './store.js';
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
      position: relative;
      flex: 1;
      background-color: black;
      border-radius: 5px;
      overflow: hidden;
    }
    .container {
      position: relative;
    }
    video, svg {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }
    svg { pointer-events: none; }
  `;

  constructor() {
    super();
    this.currentStream = null;
    this.sendInterval = null;
    this.currentDeviceId = null;
    this.selectedDevice = null;
    this.isStreaming = false;
    this.detections = [];
    this.selectedId = null;
  }

  connectedCallback() {
    super.connectedCallback();
    this.unsubscribeSelectedDevice = $selectedDevice.subscribe(value => this.selectedDevice = value);
    this.unsubscribeIsStreaming = $isStreaming.subscribe(value => this.isStreaming = value);
    this.unsubscribeDetections = $detections.subscribe(value => this.detections = value);
    this.unsubscribeSelectedId = $selectedId.subscribe(value => this.selectedId = value);
    this.tryAutoStart();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.unsubscribeSelectedDevice();
    this.unsubscribeIsStreaming();
    this.unsubscribeDetections();
    this.unsubscribeSelectedId();
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
    this.svg = SVG(this.svgElement); // SVG.js global
    this.updateOverlaySize();
    window.addEventListener('resize', this.updateOverlaySize);
  }

  updated(changedProperties) {
    if (changedProperties.has('detections') || changedProperties.has('selectedId')) {
      this.drawDetections();
    }
    this.updateStream();
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
      console.warn('Auto-start failed:', error);
      $isStreaming.set(false);
    }
  }

  updateStream() {
    if (this.isStreaming) {
      if (!this.currentStream || this.selectedDevice !== this.currentDeviceId) {
        this.startStream(this.selectedDevice);
      }
    } else if (this.currentStream) {
      this.stopStream();
    }
  }

  async startStream(deviceId) {
    if (this.currentStream) this.stopStream();
    const constraints = { video: { deviceId: { exact: deviceId }, width: 640, height: 480 } };
    this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.currentStream;
    this.currentDeviceId = deviceId;
    this.startSendingFrames();
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
    this.detections.forEach(det => {
      const isSelected = det.id === this.selectedId;
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
