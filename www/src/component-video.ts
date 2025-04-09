import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import { $selectedDevice, $isStreaming, $status, populateDevices } from './util-store';
import { wsSendBlob, wsCanSend } from './util-websocket';

// Import the CardsOverlay component
import './component-video-overlay-cards';



class ComponentVideo extends LitElement {

  #selectedDeviceController = new StoreController(this, $selectedDevice);
  #isStreamingController = new StoreController(this, $isStreaming);


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

  `;

  currentStream: MediaStream | null;
  currentDeviceId: string | null;
  originalWidth: number | null;
  originalHeight: number | null;
  readyPromise: Promise<void>;
  resolveReady: () => void;
  video: HTMLVideoElement;
  cardsOverlay: HTMLElement;
  sendInterval: number | null = null;

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
  }

  render() {
    return html`
      <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <cards-overlay id="cards-overlay"></cards-overlay>
        <stats-overlay/>
      </div>
    `;
  }

  firstUpdated() {
    this.video = this.shadowRoot.getElementById('video') as HTMLVideoElement;
    this.cardsOverlay = this.shadowRoot.getElementById('cards-overlay');

    this.video.addEventListener('loadedmetadata', () => {
      this.originalWidth = this.video.videoWidth;
      this.originalHeight = this.video.videoHeight;

      // Set dimensions on the cards overlay
      if (this.cardsOverlay && 'setDimensions' in this.cardsOverlay) {
        (this.cardsOverlay as any).setDimensions(this.originalWidth, this.originalHeight);
      }

      this.video.play().catch(e => console.error('Failed to play video:', e));
    });

    this.resolveReady();
  }

  async updated() {
    await this.updateStream();
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


}

customElements.define('video-container', ComponentVideo);
