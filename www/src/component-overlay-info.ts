import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import { $stats, $devices, $isStreaming, $selectedDevice, $showOverlayPolygon, $showOverlayPolygonClosed, $showOverlayXyxyxyxy, $sendPeriodMs, $sendQuality, $wsConnected } from './util-store';


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
    const fps = 1 / this.#statsController.value.processTime;
    const roundedFps = Math.round(fps * 10) / 10;
    const devices = this.#devicesController.value;
    const selectedDeviceId = this.#selectedDeviceController.value;
    const isStreaming = this.#isStreamingController.value;

    // Check if we have device labels
    const hasLabels = devices.some(device => device.label);

    const wsConnected = this.#wsConnectedController.value;

    return html`
      <div class="message-stats">
        <div class="connection-indicator ${wsConnected ? 'connected' : ''}" title="${wsConnected ? 'WebSocket Connected' : 'WebSocket Disconnected'}"></div>
        <span>sent/recv: ${this.#statsController.value.messagesSent}/${this.#statsController.value.messagesReceived}</span>
        <span>server fps: ${roundedFps}</span>
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
        </div>
      </div>
    `
  }
}
customElements.define('stats-overlay', StatsOverlay);
