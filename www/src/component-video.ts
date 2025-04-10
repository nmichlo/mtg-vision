import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import {
  $selectedDevice,
  $isStreaming,
  $status,
  $videoDimensions,
  $sendPeriodMs,
  populateDevices,
  $sendQuality
} from './util-store';
import { wsSendBlob, wsCanSend } from './util-websocket';


class ComponentVideo extends LitElement {

  #selectedDeviceController = new StoreController(this, $selectedDevice);
  #isStreamingController = new StoreController(this, $isStreaming);

  static styles = css`
    :host {
      display: block;
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
  `;

  currentStream: MediaStream | null;
  currentDeviceId: string | null;
  readyPromise: Promise<void>;
  resolveReady: () => void;
  video: HTMLVideoElement;
  sendInterval: number | null = null;

  constructor() {
    super();
    this.currentStream = null;
    this.currentDeviceId = null;
    this.readyPromise = new Promise(resolve => (this.resolveReady = resolve));
  }

  // OVERRIDES //

  connectedCallback() {
    super.connectedCallback();
    this.tryAutoStart();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.currentStream) {
      this.stopStream();
    }
  }

  firstUpdated() {
    this.video = this.shadowRoot.getElementById('video') as HTMLVideoElement;
    this.video.addEventListener('loadedmetadata', () => {
      $videoDimensions.set({
        width: this.video.videoWidth,
        height: this.video.videoHeight,
      });
      this.video.play().catch(e => console.error('Failed to play video:', e));
    });
    this.resolveReady();
  }

  async updated() {
    await this.updateStream();
  }

  // STREAM //

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
        canvas.toBlob(wsSendBlob, 'image/jpeg', $sendQuality.get());
      }
    }, $sendPeriodMs.get());
  }

  // RENDER //

  render() {
    return html`
      <cards-overlay></cards-overlay>
      <video id="video" autoplay muted playsinline></video>
    `;
  }
}

customElements.define('video-container', ComponentVideo);
