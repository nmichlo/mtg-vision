import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import { $stats, $devices, $isStreaming, $selectedDevice, $showOverlayPolygon, $showOverlayPolygonClosed, $showOverlayXyxyxyxy, $sendPeriodMs, $sendQuality, $wsConnected, $matchThreshold, populateDevices } from './util-store';
import { getWsUrl } from './util-websocket';


class StatsOverlay extends LitElement {
  #statsController = new StoreController(this, $stats);

  #selectedDeviceController = new StoreController(this, $selectedDevice);
  #devicesController = new StoreController(this, $devices);
  #isStreamingController = new StoreController(this, $isStreaming);
  #wsConnectedController = new StoreController(this, $wsConnected);

  #showOverlayPolygonController = new StoreController(this, $showOverlayPolygon);
  #showOverlayPolygonClosedController = new StoreController(this, $showOverlayPolygonClosed);
  #showOverlayXyxyxyxyController = new StoreController(this, $showOverlayXyxyxyxy);

  #sendPeriodMsController = new StoreController(this, $sendPeriodMs);
  #sendQualityController = new StoreController(this, $sendQuality);
  #matchThresholdController = new StoreController(this, $matchThreshold);

  static styles = css`
    :host {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 5px 10px;
      border-radius: 4px;
      font-size: 14px;
      z-index: 2;
      font-family: "Lucinda Grande", "Lucinda Sans Unicode", Helvetica, Arial, Verdana, sans-serif
    }
    .message-stats {
      display: flex;
      flex-direction: column;
      margin-bottom: 8px;
      position: relative;
    }
    .connection-indicator {
      position: absolute;
      top: 0;
      right: -5px;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background-color: red;
      box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
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
      align-items: center;
      gap: 8px;
    }
    .control-group {
      background: rgba(0, 0, 0, 0.7);
      border-radius: 4px;
      padding: 5px;
      margin-top: 5px;
    }
    select, button {
      background: rgba(0, 0, 0, 0.5);
      color: white;
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 4px;
      padding: 3px 6px;
      font-size: 12px;
      margin: 0;
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
    .stat {
        font-family: monospace;
        font-size: 12px;
    }
  `

  connectedCallback() {
    super.connectedCallback();
    // Set up a listener for device changes
    this.#devicesController.hostConnected();
    // Load the stored device when devices are available
    const unsubscribe = $devices.listen(devices => {
      if (devices.length > 0) {
        this.loadStoredDevice();
        unsubscribe();
      }
    });

    // Add event listener for device changes (when devices are plugged in or removed)
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener('devicechange', async () => {
        console.log('Device change detected, updating device list');
        await populateDevices();
      });
    }
  }

  #onDeviceChange(event) {
    const selectedDeviceId = event.target.value;
    $selectedDevice.set(selectedDeviceId);
    localStorage.setItem('selectedDeviceId', selectedDeviceId);
  }

  loadStoredDevice() {
    const storedDeviceId = localStorage.getItem('selectedDeviceId');
    const devices = this.#devicesController.value;

    if (devices.length === 0) return;

    // If we have a stored device and it exists in available devices, use it
    if (storedDeviceId && devices.some(device => device.deviceId === storedDeviceId)) {
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
      localStorage.setItem('selectedDeviceId', defaultDeviceId);
    }
  }

  render() {
    const round = (num, d=1) => Math.round(num * Math.pow(10, d)) / Math.pow(10, d);
    const stats = this.#statsController.value;
    const devices = this.#devicesController.value;
    const selectedDeviceId = this.#selectedDeviceController.value;
    const isStreaming = this.#isStreamingController.value;
    const wsConnected = this.#wsConnectedController.value;

    // Get the WebSocket URL for the tooltip
    const wsUrl = getWsUrl();
    const connectionStatus = wsConnected ? 'Connected' : 'Disconnected';
    const tooltipText = `WebSocket ${connectionStatus}: ${wsUrl}`;

    return html`
      <div class="message-stats">
        <div class="connection-indicator ${wsConnected ? 'connected' : ''}" title="${tooltipText}"></div>
        <span>sent/recv: ${this.#statsController.value.messagesSent} / ${this.#statsController.value.messagesReceived}</span>
        <span>proc time/period: <span class="stat">${round(stats.serverProcessTime*1000)} ms / ${round(stats.serverProcessPeriod*1000)} ms</span></span>
        <span>bytes send/recv: <span class="stat">${round(stats.serverRecvImBytes / stats.serverProcessPeriod / 1000, 0)} kbps / ${round(stats.serverSendImBytes / stats.serverProcessPeriod / 1000, 0)} kbps</span></span>
      </div>

      <div class="controls-container">
        <div class="control-row">
          <select @change=${this.#onDeviceChange} style="flex: 1">
            <optgroup label="devices">
              ${devices.length === 0 ? html`<option value="">No cameras found</option>` : ''}
              ${devices.map(device => html`
                <option value=${device.deviceId} ?selected=${device.deviceId === selectedDeviceId}>
                  ${device.label || 'Camera'}
                </option>
              `)}
            </optgroup>
          </select>
          <button @click=${() => $isStreaming.set(!isStreaming)}>
            ${isStreaming ? 'Stop' : 'Start'}
          </button>
        </div>

        <div class="control-group">
          <div class="checkbox-group">
            <div class="checkbox-item">
              <input type="checkbox" id="showOverlayPolygon" ?checked=${this.#showOverlayPolygonController.value} @change=${(e) => $showOverlayPolygon.set(e.target.checked)}>
              <label for="showOverlayPolygon">Polygon</label>
            </div>
            <div class="checkbox-item">
              <input type="checkbox" id="showOverlayPolygonClosed" ?checked=${this.#showOverlayPolygonClosedController.value} @change=${(e) => $showOverlayPolygonClosed.set(e.target.checked)}>
              <label for="showOverlayPolygonClosed">Closed</label>
            </div>
            <div class="checkbox-item">
              <input type="checkbox" id="showOverlayXyxyxyxy" ?checked=${this.#showOverlayXyxyxyxyController.value} @change=${(e) => $showOverlayXyxyxyxy.set(e.target.checked)}>
              <label for="showOverlayXyxyxyxy">Box</label>
            </div>
          </div>

          <div class="control-row">
            <label for="sendPeriod">FPS: ${Math.round(1000 / this.#sendPeriodMsController.value * 10) / 10}</label>
            <input type="range" min="1" max="60" value="10" class="slider" id="sendPeriod" @input=${(e) => $sendPeriodMs.set(1000 / e.target.value)} style="flex: 1">
          </div>

          <div class="control-row">
            <label for="quality">Quality: ${Math.round(this.#sendQualityController.value * 100)}%</label>
            <input type="range" min="10" max="100" value="50" class="slider" id="quality" @input=${(e) => $sendQuality.set(e.target.value / 100)} style="flex: 1">
          </div>

          <div class="control-row">
            <label for="matchThresh">Match Thresh: ${Math.round(this.#matchThresholdController.value * 100)}%</label>
            <input type="range" min="0" max="100" value="50" class="slider" id="matchThresh" @input=${(e) => $matchThreshold.set(e.target.value / 100)} style="flex: 1">
          </div>
        </div>
      </div>
    `
  }
}
customElements.define('stats-overlay', StatsOverlay);
