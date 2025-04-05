import { LitElement, html, css } from 'https://esm.run/lit';
import { StoreController } from 'https://esm.run/@nanostores/lit';
import { $devices, $selectedDevice, $isStreaming, $detections, $selectedId, $status } from './store.js';

class SidebarComponent extends LitElement {
  static styles = css`
    :host {
      width: 300px;
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
    button:hover { background-color: #00cc00; color: #1e1e1e; }
    #card-list { flex: 1; overflow-y: auto; }
  `;

  #devicesController = new StoreController(this, $devices);
  #selectedDeviceController = new StoreController(this, $selectedDevice);
  #isStreamingController = new StoreController(this, $isStreaming);
  #detectionsController = new StoreController(this, $detections);
  #selectedIdController = new StoreController(this, $selectedId);
  #statusController = new StoreController(this, $status);

  render() {
    return html`
<div id="controls">
        <select @change=${this.#onDeviceChange}>
          ${this.#devicesController.value.map((device, index) => html`
            <option value=${device.deviceId} ?selected=${device.deviceId === this.#selectedDeviceController.value}>
              ${device.label || `Camera ${index + 1}`}
            </option>
          `)}
        </select>
        <button @click=${this.#onToggleStreaming}>
          ${this.#isStreamingController.value ? 'Stop Streaming' : 'Start Streaming'}
        </button>
        <div id="status">${this.#statusController.value}</div>
      </div>
      <div id="card-list">
        ${this.#detectionsController.value.length === 0 ? html`<p>No cards detected</p>` : this.#detectionsController.value.map(det => html`
          <div @click=${() => $selectedId.set(det.id)} style="cursor: pointer; ${det.id === this.#selectedIdController.value ? 'border: 2px solid yellow;' : ''}">
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
    $selectedDevice.set(event.target.value);
  }

  #onToggleStreaming() {
    $isStreaming.set(!this.#isStreamingController.value);
  }

  #renderCardInfo() {
    const det = this.detections.find(d => d.id === this.selectedId);
    if (!det) return '';
    const bestMatch = det.matches[0];
    return html`
      <h3>${bestMatch.name}</h3>
      <p>Set: ${bestMatch.set_name || 'Unknown'} (${bestMatch.set_code || ''})</p>
      <img src="${bestMatch.img_uri || 'data:image/jpeg;base64,' + det.img}" style="width: 100%;">
    `;
  }
}

customElements.define('sidebar-component', SidebarComponent);
