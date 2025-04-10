import { $detections, $status, $stats, $wsConnected } from './util-store';
import {Payload} from "./types";
import {augmentDetections} from "./scryfall";

export let ws;

export const getWsUrl = (port?: number): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.hostname}:${port || 8000}/detect`;
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

  ws = new WebSocket(getWsUrl());

  ws.onopen = () => {
    $status.set('Connected to server.');
    $wsConnected.set(true);
  };

  ws.onmessage = (event) => {
    const data: Payload = JSON.parse(event.data);
    $detections.set(augmentDetections(data.detections));
    const stats = $stats.get()
    $stats.set({
      ...stats,
      messagesReceived: stats.messagesReceived + 1,
      processTime: (stats) ? 0.1 * data.process_time + 0.9 * stats.processTime : data.process_time,
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
