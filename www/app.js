// Import LitElement base class, html, and css tag functions from CDN
import { LitElement, html, css } from 'https://cdn.jsdelivr.net/npm/lit@3/index.js'; // Using full module path

// Helper function to check if SVG.js is loaded
function isSvgJsLoaded() {
    return typeof SVG === 'function';
}

// ======================================================================== //
// Component: Video Display (video-display)                               //
// ======================================================================== //
class VideoDisplay extends LitElement {
    // --- Styles ---
    static styles = css`
        :host { display: block; position: relative; background-color: black; overflow: hidden; flex: 1; min-width: 0; min-height: 200px; }
        video, svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: block; }
        svg { pointer-events: none; z-index: 1; }
         @media (max-width: 768px) { :host { height: 50vh; flex: 0 1 auto; min-height: 250px; } }
    `;

    // --- Properties (Inputs from Parent) ---
    static properties = {
        stream: { type: Object },
        detections: { type: Array },
        selectedId: { type: Number, nullable: true },
    };

    // --- Private Internal Fields ---
    #svgInstance = null; // Instance of SVG.js library
    #videoElement = null; // Reference to the video element
    #resizeObserver = null; // For observing container size changes

    constructor() {
        super();
        this.stream = null;
        this.detections = [];
        this.selectedId = null;
    }

    connectedCallback() {
        super.connectedCallback();
        // Use arrow function for observer callback to maintain `this` context
        this.#resizeObserver = new ResizeObserver(() => this.#updateOverlaySize());
        this.#resizeObserver.observe(this);
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.#resizeObserver?.disconnect(); // Use optional chaining
        this.#svgInstance?.remove(); // Clean up SVG.js instance if it exists
        this.#svgInstance = null;
    }

    firstUpdated() {
        // Get reference after first render
        this.#videoElement = this.shadowRoot.getElementById('video');

        if (!this.#videoElement) {
             console.error("Video element not found in shadow DOM.");
             return; // Cannot proceed without video element
        }

        // Initialize SVG.js if library is loaded
        if (isSvgJsLoaded()) {
            const svgElement = this.shadowRoot.getElementById('overlay');
            if (svgElement) {
                try {
                    this.#svgInstance = SVG(svgElement); // Attach to the SVG element
                    this.#updateOverlaySize(); // Set initial size/viewbox
                } catch (e) {
                    console.error("Failed to initialize SVG.js:", e);
                    this.#svgInstance = null; // Ensure it's null on failure
                }
            } else {
                 console.error("SVG overlay element not found in shadow DOM.");
            }
        } else {
            console.error("SVG.js library not loaded or failed to initialize globally.");
        }

        // Emit event notifying parent video element is ready (only if found)
        this.dispatchEvent(new CustomEvent('video-element-ready', {
            detail: { videoElement: this.#videoElement },
            bubbles: true,
            composed: true
        }));

        // Listen for video metadata loading to update overlay size (clean up listener in disconnected?)
        // Using {once: true} avoids manual cleanup if it only needs to run once per load
        this.#videoElement.addEventListener('loadedmetadata', this.#updateOverlaySize, { once: true });
    }

    // Use arrow function property for handler to maintain `this` context
    #updateOverlaySize = () => {
        // Guard against running if elements aren't ready or component is disconnected
        if (!this.#videoElement || !this.#svgInstance || !this.isConnected) return;

        const { videoWidth, videoHeight } = this.#videoElement;
        if (videoWidth > 0 && videoHeight > 0) {
            try {
                this.#svgInstance.viewbox(0, 0, videoWidth, videoHeight);
            } catch (e) { console.error("Error setting viewbox:", e); }
        }
    }

    updated(changedProperties) {
        if (changedProperties.has('stream') && this.#videoElement) {
            this.#videoElement.srcObject = this.stream;
             // Stream change might affect video dimensions, re-check size after a frame
             requestAnimationFrame(this.#updateOverlaySize);
        }
        // Redraw ONLY if instance exists and relevant props changed
        if (this.#svgInstance && (changedProperties.has('detections') || changedProperties.has('selectedId'))) {
            this.#drawDetections();
        }
    }

    #drawDetections() {
        // Simplified check: rely on #svgInstance being correctly initialized
        if (!this.#svgInstance) return;

        try {
            this.#svgInstance.clear();
        } catch (e) { console.error("Error clearing SVG:", e); return; }

        (this.detections ?? []).forEach(det => { // Use nullish coalescing for default
            // Use optional chaining for safer access
            if (!det?.points || !Array.isArray(det.points)) return;

            const isSelected = det.id === this.selectedId;
            const strokeColor = isSelected ? 'var(--selection-color, yellow)' : (det.color ?? 'var(--accent-color, lime)');
            const strokeWidth = isSelected ? 4 : 2;

            const validPoints = det.points.filter(p => Array.isArray(p) && p.length === 2 && !isNaN(p[0]) && !isNaN(p[1]));
            if (validPoints.length < 3) return;
            const pointsString = validPoints.map(p => p.join(',')).join(' ');

            try {
                // No need to check typeof SVG here again
                const polygon = this.#svgInstance.polygon(pointsString)
                    .fill('none')
                    .stroke({ color: strokeColor, width: strokeWidth })
                    .attr('pointer-events', 'auto')
                    .on('click', (event) => {
                        event.stopPropagation();
                        this.#dispatchSelectionEvent(det.id);
                    });

                const bestMatch = det.matches?.[0]; // Optional chaining
                if (bestMatch?.name) { // Optional chaining
                    const topPoint = validPoints.reduce((a, b) => a[1] < b[1] ? a : b);
                    if (topPoint) {
                        this.#svgInstance.text(bestMatch.name)
                            .move(topPoint[0], topPoint[1] - 15)
                            .font({ fill: 'var(--text-primary, white)', size: 12, family: 'var(--font-family)', anchor: 'start' })
                            .attr('pointer-events', 'none');
                    }
                }
            } catch (svgError) {
                console.error(`Error drawing SVG for detection ID ${det.id}:`, svgError, det);
            }
        });
    }

    // Helper to dispatch event
    #dispatchSelectionEvent(detectionId) {
         this.dispatchEvent(new CustomEvent('detection-selected', {
             detail: { detectionId },
             bubbles: true,
             composed: true
         }));
    }

    render() {
        return html`
            <video id="video" autoplay muted playsinline></video>
            <svg id="overlay"></svg>
        `;
    }
}
customElements.define('video-display', VideoDisplay);


// ======================================================================== //
// Component: Stream Controls (stream-controls)                           //
// ======================================================================== //
class StreamControls extends LitElement {
    // --- Styles (unchanged) ---
    static styles = css`
        :host { display: block; padding: 10px; margin-bottom: 15px; }
        select, button { width: 100%; padding: 10px; margin: 5px 0; border-radius: var(--border-radius, 5px); font-family: var(--font-family); font-size: 16px; cursor: pointer; }
        select { background-color: var(--background-light, #333); color: var(--text-primary, #fff); border: 2px solid var(--accent-color, #00cc00); }
        button { background-color: transparent; color: var(--accent-color, #00cc00); border: 2px solid var(--accent-color, #00cc00); transition: background-color 0.3s, color 0.3s; }
        button:hover:not(:disabled) { background-color: var(--accent-color, #00cc00); color: var(--background-dark, #1e1e1e); }
        button:disabled { opacity: 0.6; cursor: not-allowed; border-color: var(--text-secondary, #ccc); color: var(--text-secondary, #ccc); }
        #status { margin-top: 10px; font-size: 14px; color: var(--text-secondary, #ccc); min-height: 1.2em; }
        .error { color: var(--error-color, red); margin-top: 5px; font-size: 14px; }
    `;

    // --- Properties ---
    static properties = {
        videoDevices: { type: Array },
        currentDeviceId: { type: String, nullable: true },
        isStreaming: { type: Boolean },
        statusMessage: { type: String },
        error: { type: String, nullable: true },
        canStart: { type: Boolean }
    };

    // --- Event Dispatch Helpers ---
    #dispatch(eventName, detail = {}) {
        this.dispatchEvent(new CustomEvent(eventName, { detail, bubbles: true, composed: true }));
    }

    // Use arrow functions for inline event handlers to preserve `this`
    #handleDeviceChange = (event) => {
        this.#dispatch('device-change', { deviceId: event.target.value });
    }

    #handleStartStopClick = () => {
        this.#dispatch(this.isStreaming ? 'stop-stream' : 'start-stream');
    }

    render() {
        return html`
            <select
                id="select"
                @change=${this.#handleDeviceChange}
                ?disabled=${this.videoDevices.length === 0 || this.isStreaming}
                .value=${this.currentDeviceId ?? ''} >
                ${this.videoDevices.length === 0
                    ? html`<option value="">No cameras found</option>`
                    : this.videoDevices.map(device => html`
                        <option value=${device.deviceId}>
                            ${device.label || `Camera ${this.videoDevices.indexOf(device) + 1}`}
                        </option>
                    `)
                }
            </select>
            <button
                id="startCamera"
                @click=${this.#handleStartStopClick}
                ?disabled=${!this.canStart}>
                ${this.isStreaming ? 'Stop Streaming' : 'Start Streaming'}
            </button>
            <div id="status">${this.statusMessage ?? '...'}</div>
            ${this.error ? html`<div class="error">${this.error}</div>` : ''}
        `;
    }
}
customElements.define('stream-controls', StreamControls);


// ======================================================================== //
// Component: Card List (card-list)                                       //
// ======================================================================== //
class CardList extends LitElement {
    // --- Styles (unchanged) ---
    static styles = css`
        :host { display: block; flex: 1; overflow-y: auto; padding: 0 10px; margin-bottom: 10px; }
        .card-list-item { display: flex; align-items: center; margin-bottom: 10px; cursor: pointer; padding: 5px; border: 2px solid transparent; border-radius: var(--border-radius, 4px); transition: background-color 0.2s, border-color 0.2s; background-color: var(--background-medium, #2e2e2e); }
        .card-list-item:hover { background-color: var(--background-light, #3a3a3a); }
        .card-list-item.selected { border-color: var(--selection-color, yellow); background-color: #444; }
        .img-container { position: relative; width: 50px; height: 70px; overflow: hidden; border-radius: 3px; flex-shrink: 0; background-color: #555; }
        img { display: block; width: 100%; height: auto; }
        .uri-img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
        .info { margin-left: 10px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-size: 14px; }
        .info strong { display: block; margin-bottom: 3px; }
        p { color: var(--text-secondary, #ccc); text-align: center; margin-top: 20px; }
    `;

    // --- Properties ---
    static properties = {
        detections: { type: Array },
        selectedId: { type: Number, nullable: true }
    };

    // --- Event Dispatch Helper ---
    #dispatchSelectionEvent(detectionId) {
         this.dispatchEvent(new CustomEvent('card-selected', {
             detail: { detectionId },
             bubbles: true,
             composed: true
         }));
    }

    render() {
        // Sort detections directly in render, doesn't mutate original prop
        const sortedDetections = [...(this.detections ?? [])].sort((a, b) => (a?.id ?? 0) - (b?.id ?? 0));

        return html`
            ${sortedDetections.length === 0
                ? html`<p>No cards detected</p>`
                : sortedDetections.map(det => {
                      const bestMatch = det.matches?.[0]; // Optional chaining
                      const isSelected = det.id === this.selectedId;
                      return html`
                          <div
                              class="card-list-item ${isSelected ? 'selected' : ''}"
                              @click=${() => this.#dispatchSelectionEvent(det.id)}>
                              <div class="img-container">
                                  <img src=${'data:image/jpeg;base64,' + det.img} alt="Detected Card Crop" loading="lazy">
                                  ${bestMatch?.img_uri // Optional chaining
                                      ? html`<img class="uri-img" src=${bestMatch.img_uri} alt=${bestMatch.name ?? 'Match'} loading="lazy">`
                                      : ''}
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
    // --- Styles (unchanged) ---
    static styles = css`
        :host { display: block; padding: 15px; background-color: var(--background-light, #3e3e3e); border-radius: var(--border-radius, 5px); margin: 0 10px 10px 10px; min-height: 100px; }
        h3 { margin: 0 0 5px 0; font-size: 1.1em; color: var(--text-primary, white); }
        p { margin: 5px 0; font-size: 0.9em; color: var(--text-secondary, #ccc); }
        img { width: 100%; height: auto; border-radius: 3px; margin-top: 10px; display: block; background-color: #555; }
        .placeholder { color: var(--text-secondary, #ccc); font-style: italic; }
    `;

    // --- Properties ---
    static properties = {
        selectedCardDetails: { type: Object, nullable: true }
    };

    render() {
        const details = this.selectedCardDetails; // No need for ?? here, handled below

        if (!details) {
            return html`<p class="placeholder">Click a card to see details.</p>`;
        }

        // Use nullish coalescing for defaults within the template
        return html`
            <h3>${details.name ?? 'Unknown Card'}</h3>
            <p>
                Set: ${details.set_name ?? 'Unknown'} (${details.set_code ?? 'N/A'})
            </p>
            ${details.img_uri
                ? html`<img src=${details.img_uri} alt=${details.name ?? 'Card image'} loading="lazy">`
                : html`<p class="placeholder">No image available</p>`}
        `;
    }
}
customElements.define('card-info', CardInfo);


// ======================================================================== //
// Component: Sidebar Container (side-bar)                                //
// ======================================================================== //
class SideBar extends LitElement {
    // --- Styles (unchanged) ---
     static styles = css`
        :host { display: flex; flex-direction: column; width: var(--sidebar-width, 300px); background-color: var(--background-medium, #2e2e2e); padding-top: 10px; overflow: hidden; flex-shrink: 0; height: 100%; max-height: 100vh; }
        ::slotted(stream-controls) { flex-shrink: 0; }
        ::slotted(card-list) { flex-grow: 1; min-height: 50px; overflow-y: auto; }
        ::slotted(card-info) { flex-shrink: 0; }
         @media (max-width: 768px) { :host { width: 100%; height: auto; max-height: 50vh; flex: 1; padding-top: 0; } }
     `;
     // Render slot for content projection
     render() { return html`<slot></slot>`; }
}
customElements.define('side-bar', SideBar);


// ======================================================================== //
// Component: Main Application (card-detector-app)                        //
// ======================================================================== //
class CardDetectorApp extends LitElement {
    // --- Styles (unchanged) ---
    static styles = css` :host { /* Styles applied globally */ } `;

    // --- State & Properties (Managed by LitElement) ---
    // Use standard properties declared in static properties for reactivity
    static properties = {
        _detections: { state: true, type: Array },
        _selectedId: { state: true, type: Number, nullable: true },
        _selectedCardDetails: { state: true, type: Object, nullable: true },
        _videoDevices: { state: true, type: Array },
        _currentDeviceId: { state: true, type: String, nullable: true },
        _isStreaming: { state: true, type: Boolean },
        _statusMessage: { state: true, type: String },
        _error: { state: true, type: String, nullable: true },
        _canStartManually: { state: true, type: Boolean },
        _currentStream: { state: true, type: Object, nullable: true },
    };

    // --- Private Class Fields for Internal Refs & Logic ---
    #ws = null; // WebSocket instance
    #reconnectTimeout = null;
    #sendInterval = null; // Interval ID for sending frames
    #isAutoStarting = true; // Flag during initial load
    #videoElement = null; // Reference from video-display component

    constructor() {
        super();
        // Initialize reactive properties
        this._detections = [];
        this._selectedId = null;
        this._selectedCardDetails = null;
        this._videoDevices = [];
        this._currentDeviceId = null;
        this._isStreaming = false;
        this._statusMessage = "Initializing...";
        this._error = null;
        this._canStartManually = false;
        this._currentStream = null;
        console.log("CardDetectorApp constructed");
    }

    connectedCallback() {
        super.connectedCallback();
        console.log("CardDetectorApp connected");
        this.#connectWebSocket();
        // Use IIAFE (Immediately Invoked Async Function Expression) for async setup
        (async () => {
            await this.#populateDevices();
            this.#attemptAutoStart();
        })();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        console.log("CardDetectorApp disconnected");
        this.#stopStreamingInternal(); // Stop stream if component is removed
        this.#ws?.close(1000, "Component disconnected"); // Optional chaining close
        clearTimeout(this.#reconnectTimeout);
        clearInterval(this.#sendInterval);
        this.#sendInterval = null;
        this.#ws = null; // Ensure WS ref is cleared
    }

    // --- WebSocket Handling (Using Private Methods) ---
    #connectWebSocket = () => { // Use arrow function property for bound method
        if (this.#ws?.readyState === WebSocket.OPEN || this.#ws?.readyState === WebSocket.CONNECTING) return;
        clearTimeout(this.#reconnectTimeout);

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/detect`;
        this._statusMessage = 'Connecting to server...';

        try {
            this.#ws = new WebSocket(wsUrl);
            this.#ws.onopen = this.#handleWsOpen; // Assign bound methods
            this.#ws.onmessage = this.#handleWsMessage;
            this.#ws.onerror = this.#handleWsError;
            this.#ws.onclose = this.#handleWsClose;
        } catch (e) {
            console.error("WebSocket constructor failed:", e);
            this._statusMessage = "Error initializing WebSocket.";
            this._error = `WebSocket URL invalid? ${e.message}`;
            this._canStartManually = false;
        }
    }

    #handleWsOpen = () => {
        console.log('WebSocket connection established');
        this._statusMessage = 'Connected to server.';
        this._error = null;
        if (this._isStreaming && !this.#sendInterval && this.#videoElement) {
            this.#startSendingFrames();
        }
    };

    #handleWsMessage = (event) => {
         try {
            const data = JSON.parse(event.data);
            const newDetections = data.detections ?? []; // Default to empty array
            this._detections = newDetections;

            if (this._selectedId !== null) {
                const currentSelected = newDetections.find(d => d.id === this._selectedId);
                this._selectedCardDetails = currentSelected?.matches?.[0] ?? null; // Combined check
                if (!currentSelected) this._selectedId = null; // Deselect if gone
            }
         } catch (e) {
             console.error("Error parsing WebSocket message:", e, event.data);
             this._statusMessage = 'Error processing server message.';
         }
    };

    #handleWsError = (error) => {
         console.error('WebSocket error:', error);
         this._statusMessage = 'WebSocket error occurred.';
         this._error = 'Connection to detection server failed.';
    };

    #handleWsClose = (event) => {
        console.log(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason}`);
        clearInterval(this.#sendInterval); // Ensure interval is cleared on any close
        this.#sendInterval = null;
        const wasConnected = !!this.#ws; // Check if we thought we were connected
        this.#ws = null;

        if (event.code !== 1000 && event.code !== 1005 && wasConnected) {
            this._statusMessage = 'Disconnected. Reconnecting...';
            this.#reconnectTimeout = setTimeout(this.#connectWebSocket, 5000);
        } else {
            this._statusMessage = 'Disconnected from server.';
            if (this._isStreaming) this._isStreaming = false; // Update state if closed while streaming
            if (this._videoDevices.length > 0) this._canStartManually = true; // Allow restart
        }
    };


    // --- Video Stream Handling (Using Private Methods) ---
    #startStream = async (deviceId) => { // Arrow function for bound method
        console.log(`Attempting to start stream for device: ${deviceId ?? 'default'}`);
        this.#stopMediaStreamTracks(); // Stop previous tracks first
        this._currentStream = null; // Clear stream state

        const constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } } };
        if (deviceId) constraints.video.deviceId = { exact: deviceId };

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this._currentStream = stream; // Update reactive property
            this._currentDeviceId = stream.getVideoTracks()[0]?.getSettings()?.deviceId ?? deviceId ?? this._videoDevices[0]?.deviceId;
            this._error = null;
            console.log("getUserMedia successful.");

            if (this.#videoElement) {
                // Use await + event listener promise for cleaner waiting
                 await new Promise(resolve => this.#videoElement.addEventListener('playing', resolve, { once: true }));
                 console.log("Stream playing, starting frame sending.");
                 this.#startSendingFrames();
                 this._isStreaming = true;
                 this._statusMessage = 'Streaming to server...';
            } else {
                console.warn("Video element ref not available yet for 'playing' listener.");
                // Frame sending will start when video element is ready if stream active
            }
            return true;

        } catch (error) {
            console.error('Error accessing camera:', error);
            let errorMsg = 'Error: Could not access camera.';
             if (error.name === 'NotAllowedError') errorMsg = 'Camera access denied.';
             else if (error.name === 'NotFoundError') errorMsg = 'No camera found.';
             else if (error.name === 'NotReadableError') errorMsg = 'Camera is already in use.';

            this._statusMessage = errorMsg;
            this._error = errorMsg;
            this._isStreaming = false;
            this._currentStream = null;
            this._canStartManually = false;
            return false;
        }
    }

    #populateDevices = async () => { // Arrow function for bound method
        console.log("Populating video devices...");
        try {
            // Quick permission check/prompt
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
            tempStream.getTracks().forEach(track => track.stop());
        } catch (e) {
             if (e.name === 'NotAllowedError') {
                this._statusMessage = "Camera access denied.";
                this._error = "Permission needed to list/use cameras.";
                this._canStartManually = false; return; // Stop if no permission
             }
             console.warn("Pre-enumeration getUserMedia failed:", e.name);
        }

        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            this._videoDevices = devices.filter(device => device.kind === 'videoinput');
            console.log(`Found ${this._videoDevices.length} video devices.`);
            if (!this._currentDeviceId && this._videoDevices.length > 0) {
                this._currentDeviceId = this._videoDevices[0].deviceId;
            }
             this._canStartManually = this._videoDevices.length > 0; // Enable start if devices found & permissions ok
             if (!this._canStartManually && !this._error) { // If no devices found specifically
                 this._statusMessage = "No video devices found.";
                 this._error = "Please connect a camera.";
             }

        } catch (error) {
            console.error("Error enumerating devices:", error);
            this._statusMessage = "Error listing cameras.";
            this._error = "Could not get camera list.";
            this._canStartManually = false;
        }
    }

    #startSendingFrames = () => { // Arrow function for bound method
        if (this.#sendInterval || !this.#videoElement || this.#videoElement.readyState < 3 /* HAVE_FUTURE_DATA */ ) return;
        if (this.#ws?.readyState !== WebSocket.OPEN) return;

        const canvas = document.createElement('canvas');
        canvas.width = this.#videoElement.videoWidth || 640;
        canvas.height = this.#videoElement.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        const frameRate = 10; // Target FPS

        console.log(`Starting frame sending interval (${frameRate} FPS).`);

        this.#sendInterval = setInterval(() => {
            if (this.#ws?.readyState === WebSocket.OPEN && this.#videoElement?.readyState >= 3 /* HAVE_FUTURE_DATA */) {
                try {
                    ctx.drawImage(this.#videoElement, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob(blob => {
                        if (blob?.size > 0 && this.#ws?.readyState === WebSocket.OPEN) {
                            this.#ws.send(blob);
                        }
                    }, 'image/jpeg', 0.5);
                } catch (drawError) { console.error("Error drawing video to canvas:", drawError); }
            } else if (this.#ws?.readyState !== WebSocket.OPEN) {
                 // Stop interval if WS closes; handled more robustly by onclose now
                 // clearInterval(this.#sendInterval); this.#sendInterval = null;
            }
        }, 1000 / frameRate);
         // Ensure streaming state is accurate
         this._isStreaming = true;
    }

    // Combined internal stop method
    #stopStreamingInternal = (updateStatus = true) => {
         console.log("Stopping streaming internals...");
         clearInterval(this.#sendInterval);
         this.#sendInterval = null;
         this.#stopMediaStreamTracks();
         this._currentStream = null;
         this._isStreaming = false;
         if(updateStatus) {
             this._statusMessage = 'Streaming stopped.';
             this._detections = [];
             this._selectedId = null;
             this._selectedCardDetails = null;
         }
         this._canStartManually = this._videoDevices.length > 0;
    }

    // Helper to just stop media tracks
    #stopMediaStreamTracks = () => {
        this._currentStream?.getTracks().forEach(track => track.stop());
    }


    #attemptAutoStart = async () => { // Arrow function for bound method
        console.log("Attempting auto-start...");
        this.#isAutoStarting = true;
        this._statusMessage = 'Attempting auto-start...';
        const success = await this.#startStream(this._currentDeviceId);
        if (success) {
            console.log("Auto-start stream acquired.");
        } else {
            console.warn('Auto-start failed.');
        }
        this.#isAutoStarting = false;
        // Button state (_canStartManually) handled by startStream/populateDevices
    }

    // --- Event Handlers from Child Components ---
    #handleVideoElementReady = (event) => {
        this.#videoElement = event.detail.videoElement;
        // If stream started before video element was ready, start sending now
        if (this._currentStream && !this.#sendInterval) {
            this.#startSendingFrames();
             if(!this._isStreaming) this._isStreaming = true; // Ensure state is correct
        }
    }

    // Combined handler for selection from video or list
    #handleSelection = (event) => {
        const newId = event.detail.detectionId;
        this._selectedId = newId;
        const selectedDetection = this._detections.find(d => d.id === newId);
        this._selectedCardDetails = selectedDetection?.matches?.[0] ?? null;
    }

    #handleDeviceChange = async (event) => {
        const newDeviceId = event.detail.deviceId;
        this._statusMessage = `Switching camera...`;
        this.#stopStreamingInternal(false); // Stop internals without clearing detections etc.
        const success = await this.#startStream(newDeviceId);
        if (!success) {
            this._statusMessage = 'Failed to switch camera.';
            this._canStartManually = this._videoDevices.length > 0;
        }
    }

    #handleStartStream = async () => {
        this.#isAutoStarting = false;
        this._statusMessage = 'Requesting camera access...';
        this._error = null;
        await this.#startStream(this._currentDeviceId);
    }

    #handleStopStream = () => {
        this.#stopStreamingInternal(); // Full stop with status update
    }

    // --- Render Method ---
    render() {
        return html`
            <video-display
                .stream=${this._currentStream}
                .detections=${this._detections}
                .selectedId=${this._selectedId}
                @video-element-ready=${this.#handleVideoElementReady}
                @detection-selected=${this.#handleSelection} ></video-display>

            <side-bar>
                <stream-controls
                    .videoDevices=${this._videoDevices}
                    .currentDeviceId=${this._currentDeviceId}
                    .isStreaming=${this._isStreaming}
                    .statusMessage=${this._statusMessage}
                    .error=${this._error}
                    .canStart=${this._canStartManually && !this.#isAutoStarting}
                    @device-change=${this.#handleDeviceChange}
                    @start-stream=${this.#handleStartStream}
                    @stop-stream=${this.#handleStopStream}
                ></stream-controls>

                <card-list
                    .detections=${this._detections}
                    .selectedId=${this._selectedId}
                    @card-selected=${this.#handleSelection} ></card-list>

                <card-info
                    .selectedCardDetails=${this._selectedCardDetails}
                ></card-info>
            </side-bar>
        `;
    }
}
customElements.define('card-detector-app', CardDetectorApp);
