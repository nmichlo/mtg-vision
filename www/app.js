import { LitElement, html, css } from 'https://esm.run/lit';
import { atom, map } from 'https://esm.run/nanostores';
import { StoreController } from 'https://esm.run/@nanostores/lit';
// Import Fabric.js - using { fabric } named import
import * as fabric from 'https://esm.sh/fabric';

// ======================================================================== //
// Application State Store (Nano Stores)                                  //
// ======================================================================== //

// Using 'map' for state values often updated/related together
export const $appStatus = map({
    isStreaming: false,
    statusMessage: "Initializing...",
    error: null, // Store error messages
    devicesAvailable: false, // Flag if any video devices are found
    permissionGranted: null, // null=unknown, true=granted, false=denied
});

// Using 'atom' for distinct pieces of data
export const $detections = atom([]); // Array of detection objects
export const $videoDevices = atom([]); // Array of MediaDeviceInfo objects
export const $currentDeviceId = atom(null); // ID of the selected video device
export const $selectedId = atom(null); // ID of the selected detection (number)
export const $selectedCardDetails = atom(null); // Details object of the selected card


// ======================================================================== //
// Component: Video Display (video-display)                               //
// Refactored to use Fabric.js and HTML Canvas for overlay                //
// Fix: Use arrow functions for callbacks to preserve 'this' context      //
// ======================================================================== //
class VideoDisplay extends LitElement {
    // --- Styles ---
    static styles = css`
        :host { display: block; position: relative; background-color: black; overflow: hidden; flex: 1; min-width: 0; min-height: 200px; }
        video, canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: block; }
        video { object-fit: cover; z-index: 0; }
        canvas { z-index: 1; }
         @media (max-width: 768px) { :host { height: 50vh; flex: 0 1 auto; min-height: 250px; } }
    `;

    // --- Properties ---
    static properties = {
        stream: { type: Object },
    };

    // --- Store Controllers ---
    #detectionsController = new StoreController(this, $detections);
    #selectedIdController = new StoreController(this, $selectedId);

    // --- Private Internal Fields ---
    #fabricCanvas = null;
    #canvasElement = null;
    #videoElement = null;
    #resizeObserver = null;
    #isTryingToPlay = false; // Flag to prevent rapid play calls

    constructor() {
        super();
        this.stream = null;
    }

    // --- Lifecycle Callbacks ---
    connectedCallback() {
        super.connectedCallback();
        this.#resizeObserver = new ResizeObserver(this.#updateOverlaySize);
        this.#resizeObserver.observe(this);
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.#resizeObserver?.disconnect();
        this.#fabricCanvas?.dispose();
        this.#fabricCanvas = null;
        if (this.#videoElement) {
             this.#videoElement.removeEventListener('loadedmetadata', this.#updateOverlaySize);
             this.#videoElement.removeEventListener('playing', this.#updateOverlaySize);
             this.#videoElement.removeEventListener('playing', this.#handleActualPlay); // Remove playing handler
        }
        this.#canvasElement = null;
        this.#videoElement = null;
    }

    firstUpdated() {
        this.#videoElement = this.shadowRoot.getElementById('video');
        this.#canvasElement = this.shadowRoot.getElementById('overlay-canvas');

        if (!this.#videoElement || !this.#canvasElement) {
            console.error("Video or Canvas element missing.");
            return;
        }

        try {
            this.#fabricCanvas = new fabric.Canvas(this.#canvasElement, {
                selection: false,
                renderOnAddRemove: false,
                backgroundColor: 'transparent',
            });

            this.#fabricCanvas.on('mouse:down', this.#handleCanvasClick);
            this.#videoElement.addEventListener('loadedmetadata', this.#updateOverlaySize);
            // Listen for playing event to know when play() actually succeeds
            this.#videoElement.addEventListener('playing', this.#handleActualPlay);
            // No longer listen for 'playing' to trigger resize, metadata is enough
            // this.#videoElement.addEventListener('playing', this.#updateOverlaySize);

        } catch (e) {
            console.error("Fabric.js init failed:", e);
            this.#fabricCanvas = null;
        }

        this.dispatchEvent(new CustomEvent('video-element-ready', {
            detail: { videoElement: this.#videoElement }, bubbles: true, composed: true }));
    }

    updated(changedProperties) {
        if (changedProperties.has('stream')) {
            if (this.#videoElement) {
                const oldStream = changedProperties.get('stream');
                if (this.stream !== oldStream) {
                    console.log("VideoDisplay: Stream property changed. Updating srcObject.");
                    this.#videoElement.srcObject = this.stream;
                    this.#isTryingToPlay = false; // Reset flag on new stream

                     // Only attempt play if the stream is valid and video isn't already playing/attempting
                     if (this.stream && this.#videoElement.paused && !this.#isTryingToPlay) {
                          console.log("VideoDisplay: Attempting to play new stream...");
                          this.#isTryingToPlay = true; // Set flag
                          this.#videoElement.play()
                            .then(() => {
                                // Playing started successfully (or will shortly)
                                // The 'playing' event listener will handle the flag reset
                                console.log("VideoDisplay: play() promise resolved.");
                            })
                            .catch(e => {
                                console.warn("VideoDisplay: play() promise rejected:", e.name, e.message);
                                this.#isTryingToPlay = false; // Reset flag on error
                            });
                     } else if (!this.stream) {
                         // If stream is explicitly set to null, pause the video
                         this.#videoElement.pause();
                     }

                    // Defer resize/redraw slightly
                    requestAnimationFrame(() => {
                         this.#updateOverlaySize();
                         this.#drawDetections();
                    });
                }
            }
        } else {
            // If only detections or selectedId changed, redraw overlay
            // Use rAF to avoid potential layout thrashing if updates are frequent
            requestAnimationFrame(this.#drawDetections);
        }
    }

    // --- Fabric.js Specific Methods ---

    // Defined as an arrow function class field to preserve 'this' context
    #updateOverlaySize = () => {
        if (!this.#videoElement || !this.#fabricCanvas || !this.isConnected) return;

        const { offsetWidth, offsetHeight } = this;
        const width = offsetWidth > 0 ? offsetWidth : 640;
        const height = offsetHeight > 0 ? offsetHeight : 480;

        if (width > 0 && height > 0 && (this.#fabricCanvas.width !== width || this.#fabricCanvas.height !== height)) {
            console.log(`VideoDisplay: Resizing canvas to ${width}x${height}`);
            this.#fabricCanvas.setWidth(width);
            this.#fabricCanvas.setHeight(height);
            this.#canvasElement.width = width;
            this.#canvasElement.height = height;
            this.#fabricCanvas.renderAll();
            // Redraw needed after resize potentially changes scaling
            // Use rAF to ensure redraw happens after potential layout changes from resize
            requestAnimationFrame(this.#drawDetections);
        } else if (width > 0 && height > 0) {
            // If size didn't change, maybe still redraw if triggered by metadata load
             requestAnimationFrame(this.#drawDetections);
        }
    }

    // Defined as an arrow function class field
    #drawDetections = () => {
        if (!this.#fabricCanvas) return;

        this.#fabricCanvas.clear();

        const detections = this.#detectionsController.value ?? [];
        const selectedId = this.#selectedIdController.value;

        // Use Fabric canvas dimensions directly
        const canvasWidth = this.#fabricCanvas.width || 1;
        const canvasHeight = this.#fabricCanvas.height || 1;
        // FIX: Use canvas dimensions as fallback for video dimensions
        const videoWidth = this.#videoElement?.videoWidth || canvasWidth;
        const videoHeight = this.#videoElement?.videoHeight || canvasHeight;

        const scaleX = videoWidth > 0 ? canvasWidth / videoWidth : 1;
        const scaleY = videoHeight > 0 ? canvasHeight / videoHeight : 1;
        const scaleMin = Math.max(0.1, Math.min(scaleX, scaleY) || 1); // Prevent scale being 0, limit min scale effect


        detections.forEach(det => {
            if (!det?.id || !det?.points || !Array.isArray(det.points)) return;

            const isSelected = det.id === selectedId;
            const strokeColor = isSelected ? 'var(--selection-color, yellow)' : (det.color ?? 'var(--accent-color, lime)');
            const strokeWidth = isSelected ? 4 : 2;

            const validPoints = det.points.filter(p => Array.isArray(p) && p.length === 2 && !isNaN(p[0]) && !isNaN(p[1]));
            if (validPoints.length < 3) return;

            const scaledPoints = validPoints.map(p => ({ x: p[0] * scaleX, y: p[1] * scaleY }));

            try {
                const polygon = new fabric.Polygon(scaledPoints, {
                    fill: 'transparent', stroke: strokeColor,
                    strokeWidth: strokeWidth / scaleMin, selectable: false, evented: true,
                    objectCaching: false, hoverCursor: 'pointer',
                });
                polygon.detectionId = det.id;
                this.#fabricCanvas.add(polygon);

                const bestMatch = det.matches?.[0];
                if (bestMatch?.name) {
                    const topPoint = scaledPoints.reduce((a, b) => a.y < b.y ? a : b);
                    if (topPoint) {
                        const text = new fabric.Text(bestMatch.name, {
                            left: topPoint.x, top: topPoint.y - (15 / scaleY),
                            fill: 'var(--text-primary, white)', fontSize: 12 / scaleMin,
                            fontFamily: 'var(--font-family)', originX: 'left', originY: 'bottom',
                            selectable: false, evented: false,
                        });
                        this.#fabricCanvas.add(text);
                    }
                }
            } catch (fabricError) { console.error(`Fabric.js Draw Error ID ${det.id}:`, fabricError); }
        });

        this.#fabricCanvas.renderAll();
    }

    // Defined as an arrow function class field
    #handleCanvasClick = (options) => {
        const clickedObject = options.target;
        if (clickedObject && typeof clickedObject.detectionId !== 'undefined') {
            this.#dispatchSelectionEvent(clickedObject.detectionId);
        }
    }

    // Defined as an arrow function class field
    #handleActualPlay = () => {
        console.log("VideoDisplay: 'playing' event received.");
        this.#isTryingToPlay = false; // Reset flag now that playing has started
        // Initial size update might be needed *after* playing starts if metadata loaded early
        this.#updateOverlaySize();
    }

    // Dispatch event helper
    #dispatchSelectionEvent(detectionId) {
        this.dispatchEvent(new CustomEvent('detection-selected', {
            detail: { detectionId }, bubbles: true, composed: true }));
    }

    // --- Render Method ---
    render() {
        return html`
            <video id="video" autoplay muted playsinline></video>
            <canvas id="overlay-canvas"></canvas> `;
    }
}
customElements.define('video-display', VideoDisplay);


// ======================================================================== //
// Component: Stream Controls (stream-controls)                           //
// ======================================================================== //
class StreamControls extends LitElement {
    // --- Styles ---
    static styles = css`
         :host { display: block; padding: 10px; margin-bottom: 15px; }
         select, button { appearance: none; -webkit-appearance: none; width: 100%; padding: 10px; margin: 5px 0; border-radius: var(--border-radius, 5px); font-family: var(--font-family); font-size: 16px; cursor: pointer; box-sizing: border-box; }
         select { background-color: var(--background-light, #333); color: var(--text-primary, #fff); border: 2px solid var(--accent-color, #00cc00); background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2300CC00%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.6-3.6%205.4-7.9%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E'); background-repeat: no-repeat; background-position: right 10px center; background-size: 10px; padding-right: 30px; }
         button { background-color: transparent; color: var(--accent-color, #00cc00); border: 2px solid var(--accent-color, #00cc00); transition: background-color 0.3s, color 0.3s; }
         button:hover:not(:disabled) { background-color: var(--accent-color, #00cc00); color: var(--background-dark, #1e1e1e); }
         button:disabled { opacity: 0.6; cursor: not-allowed; border-color: var(--text-secondary, #ccc); color: var(--text-secondary, #ccc); background-color: transparent; }
         select:disabled { opacity: 0.6; cursor: not-allowed; border-color: var(--text-secondary, #ccc); background-image: none; }
         #status { margin-top: 10px; font-size: 14px; color: var(--text-secondary, #ccc); min-height: 1.2em; word-wrap: break-word; }
         .error { color: var(--error-color, red); margin-top: 5px; font-size: 14px; word-wrap: break-word; }
    `;
    // --- Store Controllers ---
    #devicesController = new StoreController(this, $videoDevices);
    #currentIdController = new StoreController(this, $currentDeviceId);
    #statusController = new StoreController(this, $appStatus);
    // --- Event Dispatch Helper ---
    #dispatch(eventName, detail = {}) { this.dispatchEvent(new CustomEvent(eventName, { detail, bubbles: true, composed: true })); }
    // --- Event Handlers (Arrow functions for automatic binding) ---
    #handleDeviceChange = (event) => { this.#dispatch('device-change', { deviceId: event.target.value }); }
    #handleStartStopClick = () => { this.#dispatch(this.#statusController.value.isStreaming ? 'stop-stream' : 'start-stream'); }
    // --- Render Method ---
    render() {
        const devices = this.#devicesController.value ?? [];
        const currentDeviceId = this.#currentIdController.value;
        const { isStreaming, statusMessage, error, devicesAvailable, permissionGranted } = this.#statusController.value;
        const canStart = devicesAvailable && permissionGranted !== false;
        const startButtonDisabled = !canStart || isStreaming;
        const selectDisabled = devices.length === 0 || isStreaming;

        return html`
            <select id="select" @change=${this.#handleDeviceChange} ?disabled=${selectDisabled} .value=${currentDeviceId ?? ''} >
                ${!devicesAvailable && permissionGranted === false
                    ? html`<option value="">Camera access denied</option>`
                : devices.length === 0
                    ? html`<option value="">${permissionGranted === null ? 'Checking cameras...' : 'No cameras found'}</option>`
                : devices.map((device, index) => html` <option value=${device.deviceId}> ${device.label || `Camera ${index + 1}`} </option> `)
                }
            </select>
            <button id="startCamera" @click=${this.#handleStartStopClick} ?disabled=${startButtonDisabled}> ${isStreaming ? 'Stop Streaming' : 'Start Streaming'} </button>
            <div id="status">${statusMessage ?? '...'}</div>
            ${error ? html`<div class="error">${error}</div>` : ''}
        `;
     }
}
customElements.define('stream-controls', StreamControls);


// ======================================================================== //
// Component: Card List (card-list)                                       //
// ======================================================================== //
class CardList extends LitElement {
    // --- Styles ---
    static styles = css`
        :host { display: block; flex: 1; overflow-y: auto; padding: 0 10px; margin-bottom: 10px; }
        .card-list-item { display: flex; align-items: center; margin-bottom: 10px; cursor: pointer; padding: 5px; border: 2px solid transparent; border-radius: var(--border-radius, 4px); transition: background-color 0.2s, border-color 0.2s; background-color: var(--background-medium, #2e2e2e); }
        .card-list-item:hover { background-color: var(--background-light, #3a3a3a); }
        .card-list-item.selected { border-color: var(--selection-color, yellow); background-color: #444; }
        .img-container { position: relative; width: 50px; height: 70px; overflow: hidden; border-radius: 3px; flex-shrink: 0; background-color: #555; }
        img { display: block; width: 100%; height: auto; }
        .uri-img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
        .info { margin-left: 10px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-size: 14px; color: var(--text-primary, white); }
        .info strong { display: block; margin-bottom: 3px; font-weight: bold; }
        p.empty-message { color: var(--text-secondary, #ccc); text-align: center; margin-top: 20px; font-style: italic; }
    `;
    // --- Store Controllers ---
    #detectionsController = new StoreController(this, $detections);
    #selectedIdController = new StoreController(this, $selectedId);
    // --- Event Dispatch Helper ---
    #dispatchSelectionEvent(detectionId) { this.dispatchEvent(new CustomEvent('card-selected', { detail: { detectionId }, bubbles: true, composed: true })); }
    // --- Render Method ---
    render() {
        const detections = this.#detectionsController.value ?? [];
        const selectedId = this.#selectedIdController.value;
        const sortedDetections = [...detections].sort((a, b) => (a?.id ?? 0) - (b?.id ?? 0));

        return html`
            ${sortedDetections.length === 0
                ? html`<p class="empty-message">No cards detected</p>`
                : sortedDetections.map(det => {
                      if (!det || typeof det.id === 'undefined') return null;
                      const bestMatch = det.matches?.[0];
                      const isSelected = det.id === selectedId;
                      return html`
                          <div class="card-list-item ${isSelected ? 'selected' : ''}" @click=${() => this.#dispatchSelectionEvent(det.id)} title=${`ID: ${det.id} - ${bestMatch?.name ?? 'Unknown'}`} >
                              <div class="img-container">
                                  <img src=${'data:image/jpeg;base64,' + det.img} alt="Card Crop" loading="lazy">
                                  ${bestMatch?.img_uri ? html`<img class="uri-img" src=${bestMatch.img_uri} alt=${bestMatch.name ?? 'Best Match'} loading="lazy">` : ''}
                              </div>
                              <div class="info">
                                  <strong>ID: ${det.id ?? 'N/A'}</strong>
                                  ${bestMatch?.name ?? 'Unknown'}
                              </div>
                          </div>`;
                  })
            }
        `;
    }
}
customElements.define('card-list', CardList);


// ======================================================================== //
// Component: Card Info (card-info)                                       //
// ======================================================================== //
class CardInfo extends LitElement {
    // --- Styles ---
    static styles = css`
         :host { display: block; padding: 15px; background-color: var(--background-light, #3e3e3e); border-radius: var(--border-radius, 5px); margin: 0 10px 10px 10px; min-height: 100px; }
         h3 { margin: 0 0 5px 0; font-size: 1.1em; color: var(--text-primary, white); }
         p { margin: 5px 0; font-size: 0.9em; color: var(--text-secondary, #ccc); }
         img { width: 100%; height: auto; border-radius: 3px; margin-top: 10px; display: block; background-color: #555; }
         .placeholder { color: var(--text-secondary, #ccc); font-style: italic; text-align: center; padding-top: 20px; }
    `;
    // --- Store Controller ---
    #detailsController = new StoreController(this, $selectedCardDetails);
    // --- Render Method ---
    render() {
        const details = this.#detailsController.value;
        if (!details) { return html`<p class="placeholder">Click a card to see details.</p>`; }
        return html`
            <h3>${details.name ?? 'Unknown Card'}</h3>
            <p>Set: ${details.set_name ?? 'Unknown Set'} (${details.set_code || 'N/A'})</p>
            ${details.img_uri ? html`<img src=${details.img_uri} alt=${details.name ?? 'Card image'} loading="lazy">` : html`<p class="placeholder">(No official image available)</p>` }
        `;
    }
}
customElements.define('card-info', CardInfo);


// ======================================================================== //
// Component: Sidebar Container (side-bar)                                //
// ======================================================================== //
class SideBar extends LitElement {
    // --- Styles ---
    static styles = css`
        :host { display: flex; flex-direction: column; width: var(--sidebar-width, 300px); background-color: var(--background-medium, #2e2e2e); padding-top: 10px; overflow: hidden; flex-shrink: 0; height: 100vh; max-height: 100vh; }
        ::slotted(stream-controls) { flex-shrink: 0; }
        ::slotted(card-list) { flex-grow: 1; min-height: 50px; overflow-y: auto; }
        ::slotted(card-info) { flex-shrink: 0; }
         @media (max-width: 768px) {
             :host { width: 100%; height: auto; max-height: 50vh; flex: 1; padding-top: 0; }
             ::slotted(card-list) { min-height: 100px; }
        }
     `;
     // --- Render Method ---
     render() { return html`<slot></slot>`; }
}
customElements.define('side-bar', SideBar);


// ======================================================================== //
// Component: Main Application (card-detector-app)                        //
// Uses arrow functions for event handlers.                               //
// ======================================================================== //
class CardDetectorApp extends LitElement {
    // --- Styles ---
    static styles = css` :host { display: flex; flex-grow: 1; } `;

    // --- State Properties ---
    static properties = {
        _currentStream: { state: true },
    };

    // --- Private Class Fields ---
    #ws = null;
    #reconnectTimeout = null;
    #sendInterval = null;
    #videoElement = null;

    constructor() {
        super();
        this._currentStream = null;
        console.log("CardDetectorApp constructed (Nano Stores + Fabric.js)");
    }

    // --- Lifecycle Callbacks ---
    connectedCallback() {
        super.connectedCallback();
        this.#connectWebSocket();
        (async () => {
            await this.#populateDevices();
            // Delay auto-start slightly to allow component setup
            // setTimeout(this.#attemptAutoStart, 100);
            this.#attemptAutoStart(); // Or attempt immediately
        })();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.#stopStreamingInternal(false);
        this.#ws?.close(1000, "Component disconnected");
        clearTimeout(this.#reconnectTimeout);
        clearInterval(this.#sendInterval);
        this.#ws = null;
        this.#videoElement = null;
    }

    // --- WebSocket Handling (Mostly unchanged, using arrow funcs) ---
    #connectWebSocket = async () => {
        if (this.#ws?.readyState === WebSocket.OPEN || this.#ws?.readyState === WebSocket.CONNECTING) return;
        clearTimeout(this.#reconnectTimeout);
        $appStatus.setKey('statusMessage', 'Connecting to server...');
        $appStatus.setKey('error', null);
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/detect`;
        console.log(`Connecting WebSocket to ${wsUrl}`);
        try {
            this.#ws = new WebSocket(wsUrl);
            this.#ws.onopen = this.#handleWsOpen;
            this.#ws.onmessage = this.#handleWsMessage;
            this.#ws.onerror = this.#handleWsError;
            this.#ws.onclose = this.#handleWsClose;
        } catch (e) {
             console.error("WebSocket constructor failed:", e);
             $appStatus.setKey('statusMessage', 'WebSocket init error.');
             $appStatus.setKey('error', `WebSocket connection failed: ${e.message}`);
        }
    }

    #handleWsOpen = () => {
        console.log('WebSocket connection established');
        $appStatus.setKey('statusMessage', 'Connected to server.');
        $appStatus.setKey('error', null);
        if ($appStatus.get().isStreaming && !this.#sendInterval && this.#videoElement) {
            this.#startSendingFrames();
        }
    };

    #handleWsMessage = (event) => {
         try {
            const data = JSON.parse(event.data);
            const newDetections = data.detections ?? [];
            $detections.set(newDetections);
            const currentSelectedId = $selectedId.get();
            if (currentSelectedId !== null) {
                const selectedDetectionExists = newDetections.some(d => d.id === currentSelectedId);
                if (!selectedDetectionExists) {
                    $selectedId.set(null);
                    $selectedCardDetails.set(null);
                }
            }
         } catch (e) { console.error("WS Message Parse Error:", e, "Data:", event.data); }
    };

    #handleWsError = (error) => {
         console.error('WebSocket error event:', error);
         $appStatus.setKey('statusMessage', 'WebSocket connection error.');
    };

    #handleWsClose = (event) => {
        console.log(`WebSocket closed. Code: ${event.code}, Reason: "${event.reason}"`);
        clearInterval(this.#sendInterval);
        this.#sendInterval = null;
        const wasConnected = !!this.#ws;
        this.#ws = null;
        const shouldReconnect = event.code !== 1000 && event.code !== 1005 && wasConnected;

        if (shouldReconnect) {
             $appStatus.setKey('statusMessage', 'Disconnected. Reconnecting...');
             $appStatus.setKey('error', 'Connection lost.');
             this.#reconnectTimeout = setTimeout(this.#connectWebSocket, 5000);
        } else { $appStatus.setKey('statusMessage', 'Disconnected from server.'); }

        if ($appStatus.get().isStreaming) { $appStatus.setKey('isStreaming', false); }
        $detections.set([]); $selectedId.set(null); $selectedCardDetails.set(null);
    };

    // --- Video Stream Handling (Unchanged methods, now private arrow funcs where needed) ---
#startStream = async (deviceId) => {
        console.log(`CardDetectorApp: Attempting #startStream ${deviceId ? `for device: ${deviceId}` : '(default)'}`);
        this.#stopMediaStreamTracks(); // Stop previous stream first
        this._currentStream = null;    // Clear stream property
        $appStatus.setKey('isStreaming', false); // Set state: not streaming yet
        $appStatus.setKey('statusMessage', 'Requesting camera access...');
        $appStatus.setKey('error', null);
        const constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 10 } } };
        if (deviceId) { constraints.video.deviceId = { exact: deviceId }; }

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            console.log("CardDetectorApp: getUserMedia success.");

            // Update component state & stores
            this._currentStream = stream; // Setting this triggers VideoDisplay.updated
            $appStatus.setKey('permissionGranted', true);
            const actualDeviceId = stream.getVideoTracks()[0]?.getSettings()?.deviceId;
            $currentDeviceId.set(actualDeviceId ?? deviceId ?? $videoDevices.get()[0]?.deviceId);

            // ** REMOVED Promise and internal play() call **
            // Now we rely on VideoDisplay reacting to the _currentStream update.

            // Check if video element is ready *now* - might be needed if connection is fast
            if (this.#videoElement) {
                 // Set status - Frame sending starts after 'playing' event in VideoDisplay
                 $appStatus.setKey('statusMessage', 'Waiting for video to play...');
                 // Start sending frames only *after* video reports playing successfully
                 this.#waitForVideoPlayingAndStartFrames();

            } else {
                console.warn("CardDetectorApp: Video element ref not available immediately after getUserMedia.");
                // Status will be updated once video element is ready
                 $appStatus.setKey('statusMessage', 'Video component initializing...');
            }
            return true; // Indicate stream was acquired

        } catch (error) {
            console.error('CardDetectorApp: #startStream Error:', error.name, error.message);
             let errorMsg = 'Could not access camera.'; let userMessage = 'Error starting camera.';
             let permission = $appStatus.get().permissionGranted;
             if (error.name === 'NotAllowedError'||error.name === 'SecurityError') { errorMsg = 'Camera access denied.'; userMessage = 'Camera permission denied.'; permission = false; }
             else if (error.name === 'NotFoundError'||error.name === 'DevicesNotFoundError') { errorMsg = 'No camera found.'; userMessage = 'No compatible camera found.'; permission = null; }
             else if (error.name === 'NotReadableError') { errorMsg = 'Camera is busy/unreadable.'; userMessage = 'Camera is busy/unreadable.'; }
             else if (error.name === 'OverconstrainedError') { errorMsg = `Constraints not met: ${error.constraint}`; userMessage = 'Camera settings not supported.'; }
             else if (error.name === 'AbortError') { errorMsg = 'Request aborted.'; userMessage = 'Camera request cancelled.'; }
             else if (error.name === 'TypeError') { errorMsg = 'Invalid constraints.'; userMessage = 'Invalid camera request.'; }
             $appStatus.setKey('statusMessage', userMessage); $appStatus.setKey('error', errorMsg);
             $appStatus.setKey('isStreaming', false); if (permission !== null) $appStatus.setKey('permissionGranted', permission);
             this._currentStream = null; return false;
        }
     }

    #populateDevices = async () => { /* ... same logic ... */ }

    #startSendingFrames = () => {
        if (this.#sendInterval) { console.warn("Frame sending already active."); return; } // Prevent multiple intervals
        if (!this.#videoElement || this.#videoElement.readyState < this.#videoElement.HAVE_METADATA) {
             console.warn("Cannot start frame sending: Video element not ready."); return;
        }
        if (this.#ws?.readyState !== WebSocket.OPEN || !this._currentStream?.active) {
             console.warn("Cannot start frame sending: WS not open or stream inactive."); return;
        }

        // Ensure video is actually playing before sending frames
        if (this.#videoElement.paused) {
             console.warn("Cannot start frame sending: Video is paused.");
             // Optionally try to play again? Or wait? For now, just return.
             return;
        }

        const canvas = document.createElement('canvas');
        const width = this.#videoElement.videoWidth || 640;
        const height = this.#videoElement.videoHeight || 480;
        canvas.width = width; canvas.height = height;
        const ctx = canvas.getContext('2d');
        const frameRate = 10; const intervalMs = 1000 / frameRate;

        console.log(`CardDetectorApp: Starting frame sending (${frameRate} FPS) at ${width}x${height}`);
        this.#sendInterval = setInterval(() => {
            if (this.#ws?.readyState === WebSocket.OPEN &&
                this.#videoElement?.readyState >= this.#videoElement.HAVE_CURRENT_DATA && // Check if frame data is available
                !this.#videoElement.paused && // Ensure still playing
                this._currentStream?.active)
            {
                try {
                    ctx.drawImage(this.#videoElement, 0, 0, width, height);
                    canvas.toBlob(blob => {
                        if (blob?.size > 0 && this.#ws?.readyState === WebSocket.OPEN) { this.#ws.send(blob); }
                    }, 'image/jpeg', 0.5);
                } catch (e) { console.error("Canvas draw/toBlob Error:", e); }
            } else {
                console.warn("Stopping frame send interval due to state change (WS/Video/Stream).");
                clearInterval(this.#sendInterval); this.#sendInterval = null;
            }
        }, intervalMs);
    }

    // New helper to wait for video playing event before starting frames
    #waitForVideoPlayingAndStartFrames = () => {
        if (!this.#videoElement) return;

        const startFrames = () => {
            this.#videoElement.removeEventListener('playing', startFrames); // Clean up listener
            console.log("CardDetectorApp: Video is playing, setting status and starting frames.");
            $appStatus.setKey('statusMessage', 'Streaming to server...');
            $appStatus.setKey('isStreaming', true); // Now officially streaming
            this.#startSendingFrames();
        }

        if (!this.#videoElement.paused) {
            // Already playing
            startFrames();
        } else {
            // Wait for the 'playing' event from VideoDisplay
            this.#videoElement.addEventListener('playing', startFrames, { once: true });
        }
    }


    #stopStreamingInternal = (updateStatus = true) => {
         console.log("CardDetectorApp: Stopping streaming internals...");
         clearInterval(this.#sendInterval); this.#sendInterval = null;
         this.#stopMediaStreamTracks();
         // Setting stream to null triggers VideoDisplay to update/pause
         this._currentStream = null;
         if(updateStatus) {
             $appStatus.setKey('isStreaming', false); $appStatus.setKey('statusMessage', 'Streaming stopped.');
             $appStatus.setKey('error', null); $detections.set([]); $selectedId.set(null); $selectedCardDetails.set(null);
         }
      }
    #stopMediaStreamTracks = () => {
        // Ensure we stop tracks of the *previous* stream if _currentStream is already null
        const streamToStop = this._currentStream ?? this.#videoElement?.srcObject;
        streamToStop?.getTracks().forEach(track => track.stop());
        if (this.#videoElement) this.#videoElement.srcObject = null; // Explicitly clear srcObject
     }

    #attemptAutoStart = async () => {
        console.log("Attempting auto-start...");
        const { devicesAvailable, permissionGranted, isStreaming } = $appStatus.get();
        if (isStreaming) { console.log("Skipping auto-start: Already streaming."); return; }
        if (!devicesAvailable) { console.log("Skipping auto-start: No devices."); return; }
        if (permissionGranted !== true) { console.log(`Skipping auto-start: Permission state is ${permissionGranted}.`); return; }

        $appStatus.setKey('statusMessage', 'Auto-starting stream...');
        const success = await this.#startStream($currentDeviceId.get());
        console.log(`Auto-start ${success ? 'successful' : 'failed'}.`);
     }

    // --- Event Handlers from Child Components (Defined as arrow functions) ---
    #handleVideoElementReady = (event) => {
        console.log("CardDetectorApp: Event received - video-element-ready");
        this.#videoElement = event.detail.videoElement;
        // If stream was already acquired, ensure we try starting frames
        if (this._currentStream && !this.#sendInterval) {
            this.#waitForVideoPlayingAndStartFrames();
        }
    }

    #handleSelection = (event) => {
        const newId = event.detail.detectionId;
        console.log(`Event: selection changed to ID: ${newId}`);
        $selectedId.set(newId);
        const selectedDetection = $detections.get().find(d => d.id === newId);
        $selectedCardDetails.set(selectedDetection?.matches?.[0] ?? null);
    }

    #handleDeviceChange = async (event) => {
        const newDeviceId = event.detail.deviceId;
        console.log(`Event: device-change to ${newDeviceId}`);
        if (newDeviceId === $currentDeviceId.get()) return;
        $appStatus.setKey('statusMessage', `Switching camera...`);
        this.#stopStreamingInternal(false);
        await this.#startStream(newDeviceId);
    }

    #handleStartStream = async () => {
        console.log(`Event: start-stream`);
        $appStatus.setKey('statusMessage', 'Starting stream...');
        $appStatus.setKey('error', null);
        await this.#startStream($currentDeviceId.get());
    }

    #handleStopStream = () => {
        console.log(`Event: stop-stream`);
        this.#stopStreamingInternal(true);
    }


    // --- Render Method ---
    render() {
        return html`
            <video-display
                .stream=${this._currentStream}
                @video-element-ready=${this.#handleVideoElementReady}
                @detection-selected=${this.#handleSelection}
            ></video-display>
            <side-bar>
                <stream-controls slot="controls"
                    @device-change=${this.#handleDeviceChange}
                    @start-stream=${this.#handleStartStream}
                    @stop-stream=${this.#handleStopStream}
                ></stream-controls>
                <card-list slot="list" @card-selected=${this.#handleSelection} ></card-list>
                <card-info slot="info"></card-info>
            </side-bar>
        `;
    }
}
customElements.define('card-detector-app', CardDetectorApp);
