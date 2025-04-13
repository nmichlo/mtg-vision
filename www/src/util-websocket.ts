import { $detections, $status, $stats, $wsConnected, setDetections } from './util-store';
import {Payload} from "./types";

export let ws;

/**
 * Returns the WebSocket URL based on the current location and optional port
 */
export const getWsUrl = (): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/detect`;
  return wsUrl;
}

export function wsCanSend() {
  return ws && ws.readyState === WebSocket.OPEN;
}

/**
 * Returns the current WebSocket connection status.
 */
export function isWsConnected() {
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

// Interval ID for the connection status checker
let connectionCheckerInterval = null;

/**
 * Updates the WebSocket connection status based on the current state
 */
function updateConnectionStatus() {
  $wsConnected.set(isWsConnected());
}

/**
 * Connects to the WebSocket server and handles messages.
 */
export function connectWebSocket() {
  // Set initial connection status to false when starting a connection
  $wsConnected.set(false);

  // Clear any existing interval
  if (connectionCheckerInterval) {
    clearInterval(connectionCheckerInterval);
  }

  // Set up a periodic connection status checker
  connectionCheckerInterval = setInterval(updateConnectionStatus, 2000);

  const wsUrl = getWsUrl();
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    $status.set('Connected to server.');
    $wsConnected.set(true);
  };

  ws.onmessage = (event) => {
    const data: Payload = JSON.parse(event.data);
    setDetections(data.detections);
    const stats = $stats.get()
    const wAve = (v, ave, r=0.1) => (ave) ? ((v * r) + (ave * (1 - r))) : v;
    $stats.set({
      ...stats,
      messagesReceived: stats.messagesReceived + 1,
      serverProcessTime: wAve(data.server_process_time, stats?.serverProcessTime),
      serverProcessPeriod: wAve(data.server_process_period, stats?.serverProcessPeriod),
      serverRecvImBytes: wAve(data.server_recv_im_bytes, stats?.serverRecvImBytes),
      serverSendImBytes: wAve(data.server_send_im_bytes, stats?.serverSendImBytes),
    })
  };

  ws.onerror = () => {
    $status.set('WebSocket error occurred.');
    $wsConnected.set(false);
  };

  ws.onclose = () => {
    $status.set('Disconnected from server. Reconnecting...');
    $wsConnected.set(false);
    setTimeout(connectWebSocket, 5000);
  };
}
