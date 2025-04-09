
import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import { $devices, $isStreaming, $selectedDevice, $showOverlayPolygon, $showOverlayPolygonClosed, $showOverlayXyxyxyxy, $sendPeriodMs, $sendQuality } from './util-store';


class StreamController extends LitElement {

  static styles = css`
    select, button {
      padding: 10px;
      margin: 5px 0;
      border: 2px solid #00cc00;
      background-color: #333;
      color: #fff;
      width: 100%;
    }
    button:hover {
      background-color: #00cc00;
      color: #1e1e1e;
    }
  `;

  #selectedDeviceController = new StoreController(this, $selectedDevice);
  #devicesController = new StoreController(this, $devices);
  #isStreamingController = new StoreController(this, $isStreaming);

  #showOverlayPolygonController = new StoreController(this, $showOverlayPolygon);
  #showOverlayPolygonClosedController = new StoreController(this, $showOverlayPolygonClosed);
  #showOverlayXyxyxyxyController = new StoreController(this, $showOverlayXyxyxyxy);

  #sendPeriodMsController = new StoreController(this, $sendPeriodMs);
  #sendQualityController = new StoreController(this, $sendQuality);


  connectedCallback() {
    super.connectedCallback();
    this.loadStoredDevice();
  }

  #onDeviceChange(event) {
    const selectedDeviceId = event.target.value;
    $selectedDevice.set(selectedDeviceId);
    localStorage.setItem('selectedDeviceId', selectedDeviceId);
  }

  loadStoredDevice() {
    const storedDeviceId = localStorage.getItem('selectedDeviceId');
    if (storedDeviceId && this.#devicesController.value.some(device => device.deviceId === storedDeviceId)) {
      $selectedDevice.set(storedDeviceId);
    } else {
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
    const devices = this.#devicesController.value;
    const selectedDeviceId = this.#selectedDeviceController.value;
    const isStreaming = this.#isStreamingController.value;

    return html`
      <select @change=${this.#onDeviceChange}>
        <optgroup label="devices">
          ${devices ? '' : html`<option value="">Select a camera</option>`}
          ${devices.map(device => html`
            <option value=${device.deviceId} ?selected=${device.deviceId === selectedDeviceId}>
              ${device.label || 'Camera'}
            </option>
          `)}
        </optgroup>
      </select>

      <button @click=${() => $isStreaming.set(!isStreaming)}>
        ${isStreaming ? 'Stop' : 'Start'} Stream
      </button>

      <div style="margin-bottom: 32px; margin-top: 16px">
        <input type="checkbox" id="showOverlayPolygon" ?checked=${this.#showOverlayPolygonController.value} @change=${(e) => $showOverlayPolygon.set(e.target.checked)}>
        <label for="showOverlayPolygon">Polygon</label>
        <input type="checkbox" id="showOverlayPolygonClosed" ?checked=${this.#showOverlayPolygonClosedController.value} @change=${(e) => $showOverlayPolygonClosed.set(e.target.checked)}>
        <label for="showOverlayPolygonClosed">Closed Poly</label>
        <input type="checkbox" id="showOverlayXyxyxyxy" ?checked=${this.#showOverlayXyxyxyxyController.value} @change=${(e) => $showOverlayXyxyxyxy.set(e.target.checked)}>
        <label for="showOverlayXyxyxyxy">Box</label>
        <input type="range" min="1" max="60" value="10" class="slider" id="myRange" @input=${(e) => $sendPeriodMs.set(1000 / e.target.value)}>
        <label for="myRange">Send Fps: ${Math.round(1000 / this.#sendPeriodMsController.value * 10) / 10}</label>
        <input type="range" min="10" max="100" value="50" class="slider" id="myRange2" @input=${(e) => $sendQuality.set(e.target.value / 100)}>
        <label for="myRange2">Send quality %: ${this.#sendQualityController.value * 100}</label>
      </div>


    `
  }
}
customElements.define('stream-controller', StreamController);
