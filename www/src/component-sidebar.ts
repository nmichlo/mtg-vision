import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import { $detections, $selectedId } from './util-store';
import {Match} from "./types";


class ComponentSidebar extends LitElement {
  static styles = css`
    :host {
      width: 400px;
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
      display: flex;
      flex-direction: row;
      justify-content: flex-start;
      align-items: center;
    }
    .detection-item.selected {
      border: 2px solid yellow;
    }
    .detection-item-text {
        margin-left: 10px;
        color: white;
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
                <!--<img src="${det.matches[0].img_uri}" style="height: 70px;">-->
                <span class="detection-item-text">${det.matches[0]?.name || 'Unknown'}</span>
              </div>
            `)
          )
        }
      </div>

      <div id="card-info">
        ${selectedCardMatch ? this.renderMatchInfo(selectedCardMatch) : ''}
      </div>
    `;
  }

  renderMatchInfo (match: Match) {
    const data = match.all_data
    return html`
      <h3>${match.name}</h3>
      <p style="font-size: 12px">Match Score: ${match.score}</p>
      <br/>
      <p>Set: ${match.set_name || 'Unknown'} (${match.set_code || ''})</p>
      <p>Type: ${data?.type_line || 'N/A'}</p>
      <p>Price: ${data?.price ? `$${data?.price}` : 'N/A'}</p>
      <p>${data?.oracle_text || ''}</p>
      <img src="${match.img_uri}" alt="${match.name}">
    `;
  }
}

customElements.define('sidebar-component', ComponentSidebar);
