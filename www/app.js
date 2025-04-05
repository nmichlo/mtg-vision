// app.js

import './store.js';
import './websocket.js';
import './video-container.js';
import './sidebar-component.js';
import { connectWebSocket } from './websocket.js';

// Initialize WebSocket
connectWebSocket();
