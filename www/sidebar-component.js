import { LitElement, html, css } from 'https://esm.run/lit';
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

  constructor() {
    super();
    this.devices = [];
    this.selectedDevice = null;
    this.isStreaming = false;
    this.detections = [];
    this.selectedId = null;
    this.status = '';
  }

  connectedCallback() {
    super.connectedCallback();
    this.unsubscribeDevices = $devices.subscribe(value => this.devices = value);
    this.unsubscribeSelectedDevice = $selectedDevice.subscribe(value => this.selectedDevice = value);
    this.unsubscribeIsStreaming = $isStreaming.subscribe(value => this.isStreaming = value);
    this.unsubscribeDetections = $detections.subscribe(value => this.detections = value);
    this.unsubscribeSelectedId = $selectedId.subscribe(value => this.selectedId = value);
    this.unsubscribeStatus = $status.subscribe(value => this.status = value);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.unsubscribeDevices();
    this.unsubscribeSelectedDevice();
    this.unsubscribeIsStreaming();
    this.unsubscribeDetections();
    this.unsubscribeSelectedId();
    this.unsubscribeStatus();
  }

  render() {
    return html`
      <div id="controls">
        <select @change=${this.onDeviceChange}>
          ${this.devices.map((device, index) => html`
            <option value=${device.deviceId} ?selected=${device.deviceId === this.selectedDevice}>
              ${device.label || `Camera ${index + 1}`}
            </option>
          `)}
        </select>
        <button @click=${this.onToggleStreaming}>
          ${this.isStreaming ? 'Stop Streaming' : 'Start Streaming'}
        </button>
        <div id="status">${this.status}</div>
      </div>
      <div id="card-list">
        ${this.detections.length === 0 ? html`<p>No cards detected</p>` : this.detections.map(det => html`
          <div @click=${() => $selectedId.set(det.id)} style="cursor: pointer; ${det.id === this.selectedId ? 'border: 2px solid yellow;' : ''}">
            <img src="data:image/jpeg;base64,${det.img}" style="width: 50px; height: 70px;">
            <span>${det.matches[0]?.name || 'Unknown'}</span>
          </div>
        `)}
      </div>
      <div id="card-info">
        ${this.selectedId !== null ? this.renderCardInfo() : ''}
      </div>
    `;
  }

  onDeviceChange(event) {
    $selectedDevice.set(event.target.value);
  }

  onToggleStreaming() {
    $isStreaming.set(!this.isStreaming);
  }

  renderCardInfo() {
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
