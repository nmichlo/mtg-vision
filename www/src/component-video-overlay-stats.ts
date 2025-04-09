import { LitElement, html, css } from 'lit';
import { StoreController } from '@nanostores/lit';
import { $stats } from './util-store';


class StatsOverlay extends LitElement {
  #statsController = new StoreController(this, $stats);

  static styles = css`
    .stats-overlay {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 5px 10px;
      border-radius: 4px;
      font-size: 14px;
    }
    .message-stats {
      display: flex;
      flex-direction: column;
    }
  `

  render() {
    const fps = 1 / this.#statsController.value.processTime;
    const roundedFps = Math.round(fps * 10) / 10;
    return html`
      <div class="stats-overlay">
        <div class="message-stats">
          <span>sent/recv: ${this.#statsController.value.messagesSent}/${this.#statsController.value.messagesReceived}</span>
          <span>server fps: ${roundedFps}</span>
        </div>
      </div>
    `
  }
}
customElements.define('stats-overlay', StatsOverlay);
