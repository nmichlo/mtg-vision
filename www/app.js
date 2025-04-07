import { LitElement, html, css } from 'https://esm.run/lit';
import './util-store.js';
import './util-websocket.js';
import './component-video.js';
import './component-video-overlay-cards.js';
import './component-video-overlay-stats.js';
import './component-sidebar.js';
import './component-sidebar-match-info.js';
import './component-sidebar-controller.js';
import { connectWebSocket } from './util-websocket.js';

class AppContainer extends LitElement {
  static styles = css`
    * {
      box-sizing: border-box;
      font-family: 'goudy', serif;
    }
    :host {
      display: flex;
      flex-direction: row;
      flex: 1;
      width: 100%;
      height: 100%;
    }
    sidebar-component {
        width: 400px;
    }
    @media (max-width: 768px) {
      :host {
        flex-direction: column;
      }
      sidebar-component {
        width: 100%;
      }
    }
  `;

  render() {
    return html`
      <video-container></video-container>
      <sidebar-component></sidebar-component>
    `;
  }
}
customElements.define('app-container', AppContainer);

// Initialize WebSocket
connectWebSocket();
