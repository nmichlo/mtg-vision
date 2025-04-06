import { LitElement, html, css } from 'https://esm.run/lit';
import { StoreController } from 'https://esm.run/@nanostores/lit';
import { $detections, $selectedId, $devices, $isStreaming, $selectedDevice } from './store.js';


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



class MatchInfo extends LitElement {

  /**
   * @typedef {Object} CardMatch
   * @property {string} name - Name of the card
   * @property {string} set_name - Optional set name of the card
   * @property {string} set_code - Optional set code of the card
   * @property {string} img_uri - Optional URI of the card’s image
   * @property {string} type_line - Optional type line of the card
   * @property {string} oracle_text - Optional oracle text of the card
   * @property {number} price - Optional price of the card
   */

  /**
   * @param {CardMatch} match
   */
  constructor(match) {
    super();
    this.match = match;
  }

  static styles = css`
  `

  render () {
    const match = this.match;
    return html`
      <h3>${match.name}</h3>
      <p>Set: ${match.set_name || 'Unknown'} (${match.set_code || ''})</p>
      <p>Type: ${match.type_line || 'N/A'}</p>
      <p>Price: ${match.price ? `$${match.price}` : 'N/A'}</p>
      <p>${match.oracle_text || ''}</p>
      <img src="${match.img_uri}" alt="${match.name}">
    `;
  }
}
customElements.define('match-info', MatchInfo);



class SidebarComponent extends LitElement {
  static styles = css`
    :host {
      width: 350px;
      background-color: #2e2e2e;
      padding: 10px;
      display: flex;
      flex-direction: column;
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

  #onItemClick(det) {
    const currentSelectedId = this.#selectedIdController.value;
    $selectedId.set(currentSelectedId === det.id ? null : det.id);
  }

  #getSelectedCardMatch() {
    const det = this.#detectionsController.value.find(d => d.id === this.#selectedIdController.value);
    if (!det || !det.matches) {
      return null;
    }
    return det.matches[0];
  }

  render() {
    const selectedId = this.#selectedIdController.value;
    const detections = this.#detectionsController.value;
    const selectedCardMatch = this.#getSelectedCardMatch();

    return html`
      <stream-controller></stream-controller>

      <div id="card-list">
        ${
          !detections ? (
              html`<p>No cards detected</p>`
          ) : (
            detections.map(det => html`
              <div class="detection-item ${det.id === selectedId ? 'selected' : ''}" @click=${() => this.#onItemClick(det)}>
                <img src="data:image/jpeg;base64,${det.img}" style="height: 70px;">
                <span>${det.matches[0]?.name || 'Unknown'}</span>
              </div>
            `)
          )
        }
      </div>

      <div id="card-info">
        ${selectedCardMatch ? html`<match-info .match=${selectedCardMatch}></match-info>` : ''}
      </div>
    `;
  }
}

customElements.define('sidebar-component', SidebarComponent);
