import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import { $detections, $selectedId, $devices, $isStreaming, $selectedDevice } from './util-store';


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
