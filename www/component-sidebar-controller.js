
import { LitElement, html, css } from 'https://esm.run/lit';
import { StoreController } from 'https://esm.run/@nanostores/lit';
import { $devices, $isStreaming, $selectedDevice } from './util-store.js';


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
    `
  }
}
customElements.define('stream-controller', StreamController);
