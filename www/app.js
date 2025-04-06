import './store.js';
import './websocket.js';
import './video-container.js';
import './sidebar-component.js';
import { connectWebSocket } from './websocket.js';
import { LitElement, html, css } from 'https://esm.run/lit';

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
