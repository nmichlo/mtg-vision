import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import {
  $selectedDevice,
  $isStreaming,
  $status,
  $videoDimensions,
  $sendPeriodMs,
  populateDevices,
  $sendQuality, $stats, $devices
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
  sendingActive: boolean = false;
  sendTimeoutId: number | null = null;

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
      // Get the stored device ID if available
      const storedDeviceId = localStorage.getItem('selectedDeviceId');

      // Set up video constraints
      const constraints = {
        video: {
          width: 640,
          height: 480
        }
      };

      // If we have a stored device ID, try to use it
      if (storedDeviceId) {
        constraints.video.deviceId = { exact: storedDeviceId };
      }

      // Request camera access - this will trigger browser permission prompt if needed
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.currentStream = stream;

      // Get the actual device ID that was used
      const deviceId = stream.getVideoTracks()[0].getSettings().deviceId;

      // Update the selected device in the store
      $selectedDevice.set(deviceId);

      // Save the device ID to localStorage
      localStorage.setItem('selectedDeviceId', deviceId);

      // Now that we have camera access, populate the device list
      // This ensures we get device labels (especially important for Safari)
      await populateDevices();

      // Set streaming state and update video element
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

      // Get the actual device ID that was used (important for Safari)
      const actualDeviceId = this.currentStream.getVideoTracks()[0].getSettings().deviceId;

      // Update the device ID in localStorage if it's different
      if (actualDeviceId !== deviceId) {
        localStorage.setItem('selectedDeviceId', actualDeviceId);
        $selectedDevice.set(actualDeviceId);
      }

      await this.readyPromise;
      this.video.srcObject = this.currentStream;
      this.currentDeviceId = actualDeviceId;
      this.startSendingFrames();

      // Refresh the device list to ensure we have labels (for Safari)
      await populateDevices();
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
    this.sendingActive = false;
    if (this.sendTimeoutId) {
      clearTimeout(this.sendTimeoutId);
      this.sendTimeoutId = null;
    }
  }

  startSendingFrames() {
    if (this.sendingActive) {
      return;
    }
    // Create canvas once
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    // Start the recursive sending process
    this.sendingActive = true;
    this.sendNextFrame(canvas, ctx);
  }

  sendNextFrame(canvas, ctx) {
    // If we're no longer active, don't schedule the next frame
    if (!this.sendingActive) {
      return;
    }
    // calculate the next delay
    const delay = Math.max(
      $sendPeriodMs.get(),
      ($stats.value.processTime ?? 0) * 1100 + 5,
    );
    // Schedule the next frame using the current period value (recursive)
    // Only send if connection is open
    this.sendTimeoutId = setTimeout(() => {
      if (wsCanSend()) {
        ctx.drawImage(this.video, 0, 0, 640, 480);
        canvas.toBlob(wsSendBlob, 'image/jpeg', $sendQuality.get());
      }
      this.sendNextFrame(canvas, ctx);
    }, delay);
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
