import { LitElement, html, css } from 'https://esm.run/lit';
import { StoreController } from 'https://esm.run/@nanostores/lit';
import { $detections, $selectedId, $devices, $isStreaming, $selectedDevice } from './store.js';

class SidebarComponent extends LitElement {
  static styles = css`
    :host {
      width: 350px;
      background-color: #2e2e2e;
      padding: 10px;
      display: flex;
      flex-direction: column;
    }
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
    #card-list {
      flex: 1;
      overflow-y: auto;
    }
    .detection-item {
      margin-bottom: 10px;
      cursor: pointer;
    }
    .detection-item.selected {
      border: 2px solid yellow;
    }
    #card-info img {
      width: 100%;
      max-width: 200px;
    }
    #card-info p {
      margin: 5px 0;
    }
  `;

  #detectionsController = new StoreController(this, $detections);
  #selectedIdController = new StoreController(this, $selectedId);
  #devicesController = new StoreController(this, $devices);
  #isStreamingController = new StoreController(this, $isStreaming);
  #selectedDeviceController = new StoreController(this, $selectedDevice);

  connectedCallback() {
    super.connectedCallback();
    this.loadStoredDevice();
  }

  render() {
    return html`
      <select @change=${this.#onDeviceChange}>
        <option value="">Select a camera</option>
        ${this.#devicesController.value.map(device => html`
          <option value=${device.deviceId} ?selected=${device.deviceId === this.#selectedDeviceController.value}>
            ${device.label || 'Camera'}
          </option>
        `)}
      </select>
      <button @click=${() => $isStreaming.set(!this.#isStreamingController.value)}>
        ${this.#isStreamingController.value ? 'Stop' : 'Start'} Stream
      </button>
      <div id="card-list">
        ${this.#detectionsController.value.length === 0 ? html`<p>No cards detected</p>` : this.#detectionsController.value.map(det => html`
          <div class="detection-item ${det.id === this.#selectedIdController.value ? 'selected' : ''}" @click=${() => {
            const currentSelectedId = this.#selectedIdController.value;
            $selectedId.set(
                currentSelectedId === det.id ? null : det.id
                // det.id
            );
          }}>
            <img src="data:image/jpeg;base64,${det.img}" style="width: 50px; height: 70px;">
            <span>${det.matches[0]?.name || 'Unknown'}</span>
          </div>
        `)}
      </div>
      <div id="card-info">
        ${this.#selectedIdController.value !== null ? this.#renderCardInfo() : ''}
      </div>
    `;
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

  #renderCardInfo() {
    const det = this.#detectionsController.value.find(d => d.id === this.#selectedIdController.value);
    if (!det) return '';
    const bestMatch = det.matches[0];
    return html`
      <h3>${bestMatch.name}</h3>
      <p>Set: ${bestMatch.set_name || 'Unknown'} (${bestMatch.set_code || ''})</p>
      <p>Type: ${bestMatch.type_line || 'N/A'}</p>
      <p>Price: ${bestMatch.price ? `$${bestMatch.price}` : 'N/A'}</p>
      <p>${bestMatch.oracle_text || ''}</p>
      <img src="${bestMatch.img_uri || 'data:image/jpeg;base64,' + det.img}" alt="${bestMatch.name}">
    `;
  }
}

customElements.define('sidebar-component', SidebarComponent);
