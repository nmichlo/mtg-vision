import { LitElement, html, css } from 'lit';
import {Match} from "./types";


class MatchInfo extends LitElement {

  match: Match;

  constructor(match: Match) {
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
