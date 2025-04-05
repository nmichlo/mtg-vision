import { $detections, $status } from './store.js';

export let ws;

/**
 * Connects to the WebSocket server and handles messages.
 */
export function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/detect`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    $status.set('Connected to server.');
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    $detections.set(data.detections);
  };

  ws.onerror = () => {
    $status.set('WebSocket error occurred.');
  };

  ws.onclose = () => {
    $status.set('Disconnected from server. Reconnecting...');
    setTimeout(connectWebSocket, 5000);
  };
}
