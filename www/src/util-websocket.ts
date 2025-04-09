import { $detections, $status, $stats } from './util-store';
import {Payload} from "./types";

export let ws;


export function wsCanSend() {
  return ws && ws.readyState === WebSocket.OPEN;
}


export function wsSendBlob(blob) {
  if (wsCanSend()) {
    ws.send(blob)
    const stats = $stats.get()
    $stats.set({...stats, messagesSent: stats.messagesSent + 1})
  } else {
    console.error('WebSocket is not open. Cannot send blob.');
  }
}

/**
 * Connects to the WebSocket server and handles messages.
 */
export function connectWebSocket(port?: number) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.hostname}:${port || 8000}/detect`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    $status.set('Connected to server.');
  };

  ws.onmessage = (event) => {
    const data: Payload = JSON.parse(event.data);
    $detections.set(data.detections);
    const stats = $stats.get()
    $stats.set({
      ...stats,
      messagesReceived: stats.messagesReceived + 1,
      processTime: (stats) ? 0.1 * data.process_time + 0.9 * stats.processTime : data.process_time,
    })
  };

  ws.onerror = () => {
    $status.set('WebSocket error occurred.');
  };

  ws.onclose = () => {
    $status.set('Disconnected from server. Reconnecting...');
    setTimeout(connectWebSocket, 5000);
  };
}
