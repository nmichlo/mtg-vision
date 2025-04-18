import { LitElement, html, css } from "lit";
import { StoreController } from "@nanostores/lit";
import {
  $stats,
  $devices,
  $isStreaming,
  $selectedDevice,
  $showOverlayPolygon,
  $showOverlayPolygonClosed,
  $showOverlayXyxyxyxy,
  $sendPeriodMs,
  $sendQuality,
  $wsConnected,
  $matchThreshold,
  populateDevices,
  $showControls, // <-- Already imported
} from "./util-store";
import { getWsUrl } from "./util-websocket";

class StatsOverlay extends LitElement {
  #statsController = new StoreController(this, $stats);

  #selectedDeviceController = new StoreController(this, $selectedDevice);
  #devicesController = new StoreController(this, $devices);
  #isStreamingController = new StoreController(this, $isStreaming);
  #wsConnectedController = new StoreController(this, $wsConnected);

  #showControlsController = new StoreController(this, $showControls); // <-- Already initialized

  #showOverlayPolygonController = new StoreController(
    this,
    $showOverlayPolygon,
  );
  #showOverlayPolygonClosedController = new StoreController(
    this,
    $showOverlayPolygonClosed,
  );
  #showOverlayXyxyxyxyController = new StoreController(
    this,
    $showOverlayXyxyxyxy,
  );

  #sendPeriodMsController = new StoreController(this, $sendPeriodMs);
  #sendQualityController = new StoreController(this, $sendQuality);
  #matchThresholdController = new StoreController(this, $matchThreshold);

  static styles = css`
    :host {
      position: absolute;
      top: 0px;
      left: 0px;
      width: 100%;
      z-index: 2;
      box-sizing: border-box; /* Add box-sizing */
    }
    .container {
      padding: 5px;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      flex-direction: row;
      gap: 8px;
    }
    .message-stats {
      display: flex;
      flex-direction: column;
      margin-bottom: 8px;
      position: relative;
      font-size: 12px;
    }
    .stat {
      font-family: monospace;
      font-size: 10px;
    }

    .connection-indicator {
      position: absolute;
      top: 0;
      right: -5px; /* Adjusted position */
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background-color: red;
      box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
      transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
    }
    .connection-indicator:hover {
      transform: scale(1.2);
      box-shadow: 0 0 8px rgba(0, 0, 0, 0.7);
    }
    .connection-indicator.connected {
      background-color: #00ff00;
    }
    .controls-container {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .control-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }
    .button-row {
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      gap: 8px;
    }
    .control-group {
      width: 164px;
    }
    select,
    button {
      background: rgba(0, 0, 0, 0.5);
      color: white;
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 4px;
      padding: 3px 6px;
      font-size: 12px;
      margin: 0;
      cursor: pointer; /* Add cursor pointer */
    }
    button:hover {
      background: rgba(255, 255, 255, 0.2);
    }
    label {
      font-size: 12px;
      margin-right: 5px;
      white-space: nowrap;
    }
    input[type="checkbox"] {
      margin: 0 3px 0 0;
    }
    input[type="range"] {
      height: 4px;
      background: rgba(255, 255, 255, 0.2);
      border-radius: 2px;
      -webkit-appearance: none;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: white;
    }
    .checkbox-group {
      display: flex;
      gap: 8px;
    }
    .checkbox-item {
      display: flex;
      align-items: center;
    }

    /* Style for the new toggle button */
    .toggle-controls-button {
      position: absolute;
      top: 5px;
      right: 5px; /* Position near the top right */
      width: 20px;
      height: 20px;
      padding: 0;
      font-size: 16px;
      line-height: 18px; /* Adjust for vertical centering */
      text-align: center;
      z-index: 3; /* Ensure it's above the container */
    }
    .pause-button-alt {
      position: absolute;
      top: 5px;
      left: 5px; /* Position near the top right */
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    // Set up a listener for device changes
    this.#devicesController.hostConnected();
    // Load the stored device when devices are available
    const unsubscribe = $devices.listen((devices) => {
      if (devices.length > 0) {
        this.loadStoredDevice();
        unsubscribe();
      }
    });

    // Add event listener for device changes (when devices are plugged in or removed)
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", async () => {
        console.log("Device change detected, updating device list");
        await populateDevices();
      });
    }
  }

  #onDeviceChange(event) {
    const selectedDeviceId = event.target.value;
    $selectedDevice.set(selectedDeviceId);
    localStorage.setItem("selectedDeviceId", selectedDeviceId);
  }

  loadStoredDevice() {
    const storedDeviceId = localStorage.getItem("selectedDeviceId");
    const devices = this.#devicesController.value;

    if (devices.length === 0) return;

    // If we have a stored device and it exists in available devices, use it
    if (
      storedDeviceId &&
      devices.some((device) => device.deviceId === storedDeviceId)
    ) {
      $selectedDevice.set(storedDeviceId);
    } else if ($selectedDevice.get() === null) {
      // Only set a default if no device is currently selected
      this.setDefaultDevice();
    }
  }

  setDefaultDevice() {
    const devices = this.#devicesController.value;
    if (devices.length > 0) {
      const defaultDeviceId = devices[0].deviceId;
      $selectedDevice.set(defaultDeviceId);
      localStorage.setItem("selectedDeviceId", defaultDeviceId);
    }
  }

  render() {
    const round = (num, d = 1) =>
      Math.round(num * Math.pow(10, d)) / Math.pow(10, d);
    const stats = this.#statsController.value;
    const devices = this.#devicesController.value;
    const selectedDeviceId = this.#selectedDeviceController.value;
    const isStreaming = this.#isStreamingController.value;
    const wsConnected = this.#wsConnectedController.value;
    const showControls = this.#showControlsController.value; // Get current state

    // Get the WebSocket URL for the tooltip
    const wsUrl = getWsUrl();
    const connectionStatus = wsConnected ? "Connected" : "Disconnected";
    const tooltipText = `WebSocket ${connectionStatus}: ${wsUrl}`;

    return html`
      <!-- Toggle Button -->
      <button
        class="toggle-controls-button"
        @click=${() => $showControls.set(!showControls)}
        title=${showControls ? "Hide Controls" : "Show Controls"}
      >
        ${showControls ? "-" : "+"}
      </button>

      <button
        class="pause-button-alt"
        @click=${() => $isStreaming.set(!isStreaming)}
        style="max-height: 24px; ${isStreaming
          ? "color: yellow"
          : "color: green"}; ${showControls
          ? "display: none;"
          : "display: flex;"}"
      >
        ${isStreaming ? "Pause" : "Start"}
        <div
          class="connection-indicator ${wsConnected ? "connected" : ""}"
          title="${tooltipText}"
        ></div>
      </button>

      <!-- Main Controls Container -->
      <div
        class="container"
        style="${showControls ? "display: flex;" : "display: none;"}"
      >
        <div class="control-group">
          <div class="control-row">
            <label for="sendPeriod"
              >FPS:
              ${Math.round((1000 / this.#sendPeriodMsController.value) * 10) /
              10}</label
            >
            <input
              type="range"
              min="1"
              max="60"
              .value=${1000 / this.#sendPeriodMsController.value}
              class="slider"
              id="sendPeriod"
              @input=${(e) => $sendPeriodMs.set(1000 / e.target.value)}
              style="flex: 1; max-width: 90px"
            />
          </div>
          <div class="control-row">
            <label for="quality"
              >Qual:
              ${Math.round(this.#sendQualityController.value * 100)}%</label
            >
            <input
              type="range"
              min="10"
              max="100"
              .value=${this.#sendQualityController.value * 100}
              class="slider"
              id="quality"
              @input=${(e) => $sendQuality.set(e.target.value / 100)}
              style="flex: 1; max-width: 90px;"
            />
          </div>
          <div class="control-row">
            <label for="matchThresh"
              >Thresh:
              ${Math.round(this.#matchThresholdController.value * 100)}%</label
            >
            <input
              type="range"
              min="0"
              max="100"
              .value=${this.#matchThresholdController.value * 100}
              class="slider"
              id="matchThresh"
              @input=${(e) => $matchThreshold.set(e.target.value / 100)}
              style="flex: 1; max-width: 90px;"
            />
          </div>
          <div class="checkbox-group">
            <div class="checkbox-item">
              <input
                type="checkbox"
                id="showOverlayPolygon"
                ?checked=${this.#showOverlayPolygonController.value}
                @change=${(e) => $showOverlayPolygon.set(e.target.checked)}
              />
              <label for="showOverlayPolygon">Poly</label>
            </div>
            <div class="checkbox-item">
              <input
                type="checkbox"
                id="showOverlayPolygonClosed"
                ?checked=${this.#showOverlayPolygonClosedController.value}
                @change=${(e) =>
                  $showOverlayPolygonClosed.set(e.target.checked)}
              />
              <label for="showOverlayPolygonClosed">Closed</label>
            </div>
            <div class="checkbox-item">
              <input
                type="checkbox"
                id="showOverlayXyxyxyxy"
                ?checked=${this.#showOverlayXyxyxyxyController.value}
                @change=${(e) => $showOverlayXyxyxyxy.set(e.target.checked)}
              />
              <label for="showOverlayXyxyxyxy">Box</label>
            </div>
          </div>
        </div>

        <div class="button-row">
          <select
            @change=${this.#onDeviceChange}
            style="flex: 1; max-width: 70px; max-height: 24px"
          >
            <optgroup label="devices">
              ${devices.length === 0
                ? html`<option value="">No cameras found</option>`
                : ""}
              ${devices.map(
                (device) => html`
                  <option
                    value=${device.deviceId}
                    ?selected=${device.deviceId === selectedDeviceId}
                  >
                    ${device.label || "Camera"}
                  </option>
                `,
              )}
            </optgroup>
          </select>
          <button
            @click=${() => $isStreaming.set(!isStreaming)}
            style="max-height: 24px; ${isStreaming
              ? "color: yellow"
              : "color: green"}"
          >
            ${isStreaming ? "Pause" : "Start"}
          </button>
        </div>

        <div class="message-stats">
          <div
            class="connection-indicator ${wsConnected ? "connected" : ""}"
            title="${tooltipText}"
          ></div>
          <span
            >sent/recv: ${this.#statsController.value.messagesSent} /
            ${this.#statsController.value.messagesReceived}</span
          >
          <span
            >sent/recv:
            <span class="stat"
              >${round(
                stats.serverRecvImBytes / stats.serverProcessPeriod / 1000,
                0,
              )}
              /
              ${round(
                stats.serverSendImBytes / stats.serverProcessPeriod / 1000,
                0,
              )}
              kbps</span
            ></span
          >
          <span
            >proc:
            <span class="stat"
              >${round(stats.serverProcessTime * 1000)} /
              ${round(stats.serverProcessPeriod * 1000)} ms</span
            ></span
          >
        </div>
      </div>
    `;
  }
}
customElements.define("stats-overlay", StatsOverlay);
