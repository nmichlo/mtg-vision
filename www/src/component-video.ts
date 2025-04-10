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
      // First populate the devices list - this will handle Safari permissions
      await populateDevices();

      // Check if we have a stored device ID
      const storedDeviceId = localStorage.getItem('selectedDeviceId');
      let deviceConstraints = { width: 640, height: 480 };
      let useStoredDevice = false;

      // Get the current list of devices
      const availableDevices = $devices.get();

      // If we have a stored device and it's in the available devices, use it
      if (storedDeviceId && availableDevices.some(device => device.deviceId === storedDeviceId)) {
        deviceConstraints = {
          deviceId: { exact: storedDeviceId },
          width: 640,
          height: 480
        };
        console.log('Using stored device:', storedDeviceId);
        useStoredDevice = true;
      }

      // Get the stream with the appropriate constraints
      const stream = await navigator.mediaDevices.getUserMedia({ video: deviceConstraints });
      this.currentStream = stream;

      // Get the actual device ID that was used
      const deviceId = stream.getVideoTracks()[0].getSettings().deviceId;

      // Set the selected device
      $selectedDevice.set(deviceId);

      // Save to localStorage if we didn't use the stored device or if it's different
      if (!useStoredDevice || storedDeviceId !== deviceId) {
        console.log('Saving new device ID to localStorage:', deviceId);
        localStorage.setItem('selectedDeviceId', deviceId);
      }

      // After getting a stream, refresh the device list again to ensure we have all devices with labels
      // This is especially important for Safari
      if (availableDevices.some(device => !device.label)) {
        console.log('Refreshing device list after stream access');
        await populateDevices();
      }

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
