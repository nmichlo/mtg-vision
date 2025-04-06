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
        video-container {
          flex: 1;
        }
        sidebar-component {
          width: 300px;
        }
        @media (max-width: 768px) {
          body {
            flex-direction: column;
          }
          sidebar-component {
            width: 100%;
          }
        }
    `;

    render() {
        return html`
            <body>
                <video-container></video-container>
                <sidebar-component></sidebar-component>
            </body>
        `;
    }
}
customElements.define('app-container', AppContainer);

// Initialize WebSocket
connectWebSocket();
