import { LitElement, html, css } from 'lit';
import './util-store';
import './util-websocket';
import './component-video';
import './component-video-overlay-cards';
import './component-video-overlay-stats';
import './component-sidebar';
import './component-sidebar-controller';
import { connectWebSocket } from './util-websocket';
import {fetchSymbology} from "./scryfall";

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


// Initialize WebSocket connection and fetch Scryfall symbology
connectWebSocket();
fetchSymbology();
