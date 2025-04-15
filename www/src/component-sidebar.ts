import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import {$detections, $matchThreshold, $selectedId, getDetections} from './util-store';
import {Detection, Match} from "./types";


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
      min-height: 100px;
    }

    @media (max-width: 768px) {
        #card-list {
          display: none;
        }
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
    .detection-item-info {
        margin-left: 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
    }
    .detection-item-text {
        color: white;
    }
    .detection-item-score {
        font-size: 9px;
        color: #c0ffee;
    }
    .detection-item-img {
        height: 80px;
    }
    .all-card-matches {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: stretch;
        overflow: auto;
        max-height: 600px;
    }
    .card-match img {
      width: 100%;
      max-width: 200px;
    }
    .card-match p {
      margin: 5px 0;
      color: white;
    }

    .oracle-text-container {
      max-height: 120px;
      margin: 10px 0 8px;
    }
    pre.oracle-text {
      font-size: 12px;
      white-space: pre-wrap;
      overflow: auto;
      max-height: 150px;
      background-color: #1e1e1e;
      padding: 8px;
      border-radius: 4px;
      color: white;
      margin: 0;
    }
    .total-cost {
      font-weight: bold;
      color: #00cc00;
      margin-top: 10px;
      text-align: right;
    }
  `;

  // don't remove even though unused
  #thresholdController = new StoreController(this, $matchThreshold);
  #detectionsController = new StoreController(this, $detections);
  #selectedIdController = new StoreController(this, $selectedId);

  connectedCallback() {
    super.connectedCallback();
  }

  #onItemClick(det) {
    const currentSelectedId = this.#selectedIdController.value;
    $selectedId.set(currentSelectedId === det.id ? null : det.id);
  }

  #getSelectedCard(): Detection | undefined {
    const det = getDetections().find(d => d.id === this.#selectedIdController.value);
    if (!det) {
      return null;
    }
    return det;
  }

  #getSelectedCardMatches(): Match[] {
    const det = this.#getSelectedCard();
    if (!det || !det.matches) {
      return [];
    }
    const matches = det?.matches || [];
    // sort matches by score, highest first
    matches.sort((a, b) => b.score - a.score);
    // at most return 3 matches
    return matches.slice(0, 3);
  }

  #calculateTotalCost(detections) {
    if (!detections || detections.length === 0) return 0;

    let totalCost = 0;
    detections.forEach(det => {
      const match = det.matches?.[0];
      if (match?.all_data?.prices?.usd) {
        const price = parseFloat(match.all_data.prices.usd);
        if (!isNaN(price)) {
          totalCost += price;
        }
      }
    });

    return totalCost.toFixed(2);
  }

  render() {
    const selectedId = this.#selectedIdController.value;
    const detections = getDetections(); // this.#detectionsController.value;
    const totalCost = this.#calculateTotalCost(detections);

    return html`
      ${detections && detections.length > 0 ? html`
        <div class="total-cost">Total Value: $${totalCost}</div>
      ` : ''}

      <div id="card-list">
        ${
          !detections ? (
              html`<p>No cards detected</p>`
          ) : (
            detections.map(det => {
              const match = det.matches[0];
              return html`
                <div class="detection-item ${det.id === selectedId ? 'selected' : ''}" @click=${() => this.#onItemClick(det)}>
                  <img src="data:image/jpeg;base64,${det.img}" class="detection-item-img">
                    <div class="detection-item-info">
                        <span class="detection-item-text">${match?.name || 'Unknown'} ${match?.extra_data?.mana_cost ?? ''} $${match?.all_data?.prices?.usd || '0'}</span>
                        <span class="detection-item-score">score: ${match?.score || '0'}</span>
                    </div>
                </div>
                `
            })
          )
        }
      </div>

      <div class="all-card-matches">
        ${this.#getSelectedCardMatches().map(match => html`
            <div class="card-match">
                ${this.renderMatchInfo(match)}
            </div>
        `)}
      </div>
    `;
  }

  renderMatchInfo (match: Match) {
    const data = match.all_data;
    const formattedOracleText = match?.extra_data?.oracle_text ?? '';
    const formattedCost = match?.extra_data?.mana_cost ?? '';
    return html`
      <hr style="border: 1px solid #444; margin: 10px 0;">
      <h3>${match.name} ${formattedCost}</h3>
      <p style="font-size: 14px; color: #c0ffee;">Match Score: ${match.score}</p>
      <p><span style="color: yellow;">Set:</span> ${match.set_name || 'Unknown'} (${match.set_code || ''})</p>
      <p><span style="color: yellow;">Type:</span> ${data?.type_line || 'N/A'}</p>
      <p><span style="color: yellow;">Price:</span> ${data?.prices?.usd ? `$${data.prices.usd}` : 'N/A'}</p>
      <div class="oracle-text-container">
        <pre class="oracle-text">${formattedOracleText}</pre>
      </div>
      <img src="${match.img_uri}" alt="${match.name}">
<!--      <img src="data:image/jpeg;base64,${this.#getSelectedCard().img}" alt="${match.name}">-->
    `;
  }
}

customElements.define('sidebar-component', ComponentSidebar);
