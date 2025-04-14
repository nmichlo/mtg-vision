import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import {
  $selectedDevice,
  $isStreaming,
  $status,
  $videoDimensions,
  $sendPeriodMs,
  populateDevices,
  $sendQuality,
  $stats
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
      pointer-events: none; /* Keep this so overlay clicks work */
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
    // Ensure stream is fully stopped when component is removed
    this.stopStreamAndClearVideo();
  }

firstUpdated() {
  this.video = this.shadowRoot.getElementById('video') as HTMLVideoElement;

  // Existing listener for initial dimensions
  this.video.addEventListener('loadedmetadata', () => {
    $videoDimensions.set({
      width: this.video.videoWidth,
      height: this.video.videoHeight,
    });
    if ($isStreaming.get()) {
      this.video.play().catch(e => console.error('Failed to play video initially:', e));
    }
  });

  // New listener for dimension changes on rotation
  this.video.addEventListener('resize', () => {
    $videoDimensions.set({
      width: this.video.videoWidth,
      height: this.video.videoHeight,
    });
  });

  this.resolveReady();
}

  async updated() {
    await this.manageStreamState();
  }

  // STREAM MANAGEMENT //

  async tryAutoStart() {
    try {
      const storedDeviceId = localStorage.getItem('selectedDeviceId');
      const constraints = {video: {
        width: 640,
        height: 480,
        //   height: 720,
        deviceId: undefined
      }};
      if (storedDeviceId) {
        constraints.video.deviceId = { exact: storedDeviceId };
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.currentStream = stream;
      const deviceId = stream.getVideoTracks()[0].getSettings().deviceId;
      $selectedDevice.set(deviceId);
      localStorage.setItem('selectedDeviceId', deviceId);
      await populateDevices();
      $isStreaming.set(true); // Indicate streaming should start
      await this.readyPromise;
      this.video.srcObject = this.currentStream;
      this.currentDeviceId = deviceId; // Set current device ID
      // Video play is handled by loadedmetadata or manageStreamState
      this.startSendingFrames(); // Start sending frames now that stream is ready
    } catch (error) {
      console.error('Auto-start failed:', error);
      $isStreaming.set(false);
      $status.set('Camera access failed: ' + error.message);
    }
  }

  async manageStreamState() {
    const shouldBeStreaming = this.#isStreamingController.value;
    const targetDeviceId = this.#selectedDeviceController.value;

    await this.readyPromise; // Ensure video element is ready

    if (shouldBeStreaming) {
      // Start stream if it's not running OR if the device has changed
      if (!this.currentStream || targetDeviceId !== this.currentDeviceId) {
        await this.startStream(targetDeviceId); // This sets srcObject, plays, and starts sending
      }
      // If stream IS running and matches the target device, but video is paused (e.g., after clicking Start again)
      else if (this.currentStream && targetDeviceId === this.currentDeviceId && this.video.paused) {
         this.video.play().catch(e => console.error('Failed to play video on resume:', e));
         this.startSendingFrames(); // Ensure frame sending restarts
      }
    } else {
      // Pause if the stream is running and video is not already paused
      if (this.currentStream && !this.video.paused) {
         this.pauseVideoAndSending();
      }
    }
  }

  async startStream(deviceId) {
    // Stop existing stream completely before starting a new one
    if (this.currentStream) {
      this.stopStreamAndClearVideo();
    }
    try {
      const constraints = { video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
          width: 640,
          height: 480,
          // height: 720,
        }
      };
      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      const actualDeviceId = this.currentStream.getVideoTracks()[0].getSettings().deviceId;

      if (actualDeviceId !== deviceId) {
        localStorage.setItem('selectedDeviceId', actualDeviceId);
        $selectedDevice.set(actualDeviceId); // Update store if deviceId changed
      }

      await this.readyPromise;
      this.video.srcObject = this.currentStream;
      this.currentDeviceId = actualDeviceId; // Update the current device ID

      // Video play is handled by the 'loadedmetadata' event listener the first time,
      // otherwise we need to play it here if metadata is already loaded.
      if (this.video.readyState >= this.video.HAVE_METADATA) {
        this.video.play().catch(e => console.error('Failed to play video on startStream:', e));
      }

      this.startSendingFrames(); // Start sending frames
      await populateDevices(); // Refresh device list potentially with labels
    } catch (error) {
      console.error('Failed to start stream:', error);
      $status.set('Failed to start camera: ' + error.message);
      $isStreaming.set(false); // Set streaming to false on error
      this.stopStreamAndClearVideo(); // Clean up on error
    }
  }

  /** Stops sending frames and pauses the video element */
  pauseVideoAndSending() {
    this.sendingActive = false;
    if (this.sendTimeoutId) {
      clearTimeout(this.sendTimeoutId);
      this.sendTimeoutId = null;
    }
    if (this.video) {
      this.video.pause();
    }
    console.log("Video paused, frame sending stopped.");
  }

  /** Stops the MediaStream tracks and clears the video srcObject */
  stopStreamAndClearVideo() {
    this.pauseVideoAndSending(); // Ensure sending stops and video is paused first

    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop());
      console.log("MediaStream tracks stopped.");
    }
    this.currentStream = null;
    this.currentDeviceId = null; // Reset current device ID

    if (this.video) {
      this.video.srcObject = null; // Clear the video display
      console.log("Video source cleared.");
    }
  }

  // FRAME SENDING //

  startSendingFrames() {
    // Don't start if already sending or should not be streaming
    if (this.sendingActive || !$isStreaming.get()) {
      return;
    }
    console.log("Starting frame sending loop.");
    // Create canvas once if it doesn't exist or dimensions changed (though dimensions are fixed here)
    // For simplicity, recreating each time startSendingFrames is called. Could optimize.
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    // Start the recursive sending process
    this.sendingActive = true;
    this.sendNextFrame(canvas, ctx);
  }

  sendNextFrame(canvas, ctx) {
    // Stop if no longer active or streaming is turned off
    if (!this.sendingActive || !$isStreaming.get()) {
      this.sendingActive = false;
      console.log("Stopping frame sending loop.");
      return;
    }

    // calculate the next delay
    const delay = Math.max(
      $sendPeriodMs.get(),
      ($stats.value.serverProcessTime ?? 0.01) * 1000 * 1.5,
      // ($stats.value.serverProcessPeriod ?? 0.01) * 1000 * 0.8,
    );

    // Schedule the next frame capture and send
    this.sendTimeoutId = setTimeout(() => {
      const shouldSend = (
        this.sendingActive
        && $isStreaming.get()
        && wsCanSend()
        && this.video
        && this.video.readyState >= this.video.HAVE_CURRENT_DATA
      );
      // 1. send frame
      if (shouldSend) {
        try {
          ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(wsSendBlob, 'image/jpeg', $sendQuality.get());
        } catch (e) {
          console.error("Error capturing or sending frame:", e); // TODO: could stop?
        }
      }
      // 2. schedule next frame
      this.sendNextFrame(canvas, ctx);
    }, delay);
  }

  // RENDER //

  render() {
    return html`
      <cards-overlay></cards-overlay>
      <video id="video" muted playsinline></video>
    `;
  }
}

customElements.define('video-container', ComponentVideo);
