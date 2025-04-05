import { LitElement, html, css } from 'https://esm.run/lit';
import { atom, map } from 'https://esm.run/nanostores';
import { StoreController } from 'https://esm.run/@nanostores/lit';

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
// Now uses StoreController for detections and selectedId                 //
// Still receives stream via property from parent                         //
// ======================================================================== //
class VideoDisplay extends LitElement {
    static styles = css` /* ... (Styles same as previous example) ... */
        :host { display: block; position: relative; background-color: black; overflow: hidden; flex: 1; min-width: 0; min-height: 200px; }
        video, svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: block; }
        svg { pointer-events: none; z-index: 1; }
         @media (max-width: 768px) { :host { height: 50vh; flex: 0 1 auto; min-height: 250px; } }
    `;

    // --- Properties (Stream still passed as prop) ---
    static properties = {
        stream: { type: Object },
    };

    // --- Store Controllers ---
    #detectionsController = new StoreController(this, $detections);
    #selectedIdController = new StoreController(this, $selectedId);

    // --- Private Internal Fields ---
    #svgInstance = null;
    #videoElement = null;
    #resizeObserver = null;

    constructor() {
        super();
        this.stream = null; // Initialize prop
    }

    connectedCallback() {
        super.connectedCallback();
        this.#resizeObserver = new ResizeObserver(this.#updateOverlaySize);
        this.#resizeObserver.observe(this);
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.#resizeObserver?.disconnect();
        this.#svgInstance?.remove();
        this.#svgInstance = null;
    }

    firstUpdated() {
        this.#videoElement = this.shadowRoot.getElementById('video');
        if (!this.#videoElement) { console.error("Video element missing."); return; }

        const svgElement = this.shadowRoot.getElementById('overlay');
        if (svgElement) {
            try { this.#svgInstance = SVG(svgElement); this.#updateOverlaySize(); }
            catch (e) { console.error("SVG.js init failed:", e); this.#svgInstance = null; }
        } else { console.error("SVG overlay element missing."); }

        this.dispatchEvent(new CustomEvent('video-element-ready', {
            detail: { videoElement: this.#videoElement }, bubbles: true, composed: true }));

        this.#videoElement.addEventListener('loadedmetadata', this.#updateOverlaySize, { once: true });
    }

    #updateOverlaySize = () => {
        if (!this.#videoElement || !this.#svgInstance || !this.isConnected) return;
        const { videoWidth, videoHeight } = this.#videoElement;
        if (videoWidth > 0 && videoHeight > 0) {
            try { this.#svgInstance.viewbox(0, 0, videoWidth, videoHeight); }
            catch (e) { console.error("Error setting viewbox:", e); }
        }
    }

    // `updated` lifecycle still handles prop changes (like stream)
    // and triggers redraw if necessary (though StoreController also triggers updates)
    updated(changedProperties) {
        if (changedProperties.has('stream') && this.#videoElement) {
            this.#videoElement.srcObject = this.stream;
            requestAnimationFrame(this.#updateOverlaySize);
        }
        // Drawing might be triggered by StoreController *or* prop changes
        // It's safe to call draw here as StoreController ensures updates on store change
        this.#drawDetections();
    }

    #drawDetections() {
        if (!this.#svgInstance) return;
        try { this.#svgInstance.clear(); } catch (e) { return; }

        // Access store values via controller
        const detections = this.#detectionsController.value ?? [];
        const selectedId = this.#selectedIdController.value;

        detections.forEach(det => {
            if (!det?.points || !Array.isArray(det.points)) return;
            const isSelected = det.id === selectedId;
            const strokeColor = isSelected ? 'var(--selection-color, yellow)' : (det.color ?? 'var(--accent-color, lime)');
            const strokeWidth = isSelected ? 4 : 2;
            const validPoints = det.points.filter(p => Array.isArray(p) && p.length === 2 && !isNaN(p[0]) && !isNaN(p[1]));
            if (validPoints.length < 3) return;
            const pointsString = validPoints.map(p => p.join(',')).join(' ');

            try {
                const polygon = this.#svgInstance.polygon(pointsString)
                    .fill('none').stroke({ color: strokeColor, width: strokeWidth })
                    .attr('pointer-events', 'auto')
                    .on('click', (event) => { event.stopPropagation(); this.#dispatchSelectionEvent(det.id); });

                const bestMatch = det.matches?.[0];
                if (bestMatch?.name) {
                    const topPoint = validPoints.reduce((a, b) => a[1] < b[1] ? a : b);
                    if (topPoint) {
                        this.#svgInstance.text(bestMatch.name)
                            .move(topPoint[0], topPoint[1] - 15)
                            .font({ fill: 'var(--text-primary, white)', size: 12, family: 'var(--font-family)', anchor: 'start' })
                            .attr('pointer-events', 'none');
                    }
                }
            } catch (svgError) { console.error(`SVG Draw Error ID ${det.id}:`, svgError); }
        });
    }

    #dispatchSelectionEvent(detectionId) {
        this.dispatchEvent(new CustomEvent('detection-selected', {
            detail: { detectionId }, bubbles: true, composed: true }));
    }

    render() {
        // Stream is still a prop, drawing happens in updated/drawDetections
        return html`
            <video id="video" autoplay muted playsinline .srcObject=${this.stream}></video>
            <svg id="overlay"></svg>
        `;
    }
}
customElements.define('video-display', VideoDisplay);


// ======================================================================== //
// Component: Stream Controls (stream-controls)                           //
// Uses StoreController for devices, deviceId, status, error, streaming   //
// Computes `canStart` locally based on store values                      //
// ======================================================================== //
class StreamControls extends LitElement {
    static styles = css` /* ... (Styles same as previous example) ... */
         :host { display: block; padding: 10px; margin-bottom: 15px; }
        select, button { width: 100%; padding: 10px; margin: 5px 0; border-radius: var(--border-radius, 5px); font-family: var(--font-family); font-size: 16px; cursor: pointer; }
        select { background-color: var(--background-light, #333); color: var(--text-primary, #fff); border: 2px solid var(--accent-color, #00cc00); }
        button { background-color: transparent; color: var(--accent-color, #00cc00); border: 2px solid var(--accent-color, #00cc00); transition: background-color 0.3s, color 0.3s; }
        button:hover:not(:disabled) { background-color: var(--accent-color, #00cc00); color: var(--background-dark, #1e1e1e); }
        button:disabled { opacity: 0.6; cursor: not-allowed; border-color: var(--text-secondary, #ccc); color: var(--text-secondary, #ccc); }
        #status { margin-top: 10px; font-size: 14px; color: var(--text-secondary, #ccc); min-height: 1.2em; }
        .error { color: var(--error-color, red); margin-top: 5px; font-size: 14px; }
    `;

    // --- No properties needed, all come from store ---

    // --- Store Controllers ---
    #devicesController = new StoreController(this, $videoDevices);
    #currentIdController = new StoreController(this, $currentDeviceId);
    #statusController = new StoreController(this, $appStatus); // Map store

    // --- Event Dispatch Helpers ---
    #dispatch(eventName, detail = {}) {
        this.dispatchEvent(new CustomEvent(eventName, { detail, bubbles: true, composed: true }));
    }

    #handleDeviceChange = (event) => {
        this.#dispatch('device-change', { deviceId: event.target.value });
    }

    #handleStartStopClick = () => {
        const isStreaming = this.#statusController.value.isStreaming; // Get current value
        this.#dispatch(isStreaming ? 'stop-stream' : 'start-stream');
    }

    render() {
        const devices = this.#devicesController.value ?? [];
        const currentDeviceId = this.#currentIdController.value;
        const { isStreaming, statusMessage, error, devicesAvailable, permissionGranted } = this.#statusController.value;

        // Compute `canStart` based on store values
        const canStart = devicesAvailable && permissionGranted !== false;

        return html`
            <select
                id="select"
                @change=${this.#handleDeviceChange}
                ?disabled=${devices.length === 0 || isStreaming}
                .value=${currentDeviceId ?? ''}
            >
                ${devices.length === 0
                    ? html`<option value="">No cameras found</option>`
                    : devices.map(device => html`
                        <option value=${device.deviceId}>
                            ${device.label || `Camera ${devices.indexOf(device) + 1}`}
                        </option>
                    `)
                }
            </select>
            <button
                id="startCamera"
                @click=${this.#handleStartStopClick}
                ?disabled=${!canStart || isStreaming}> ${isStreaming ? 'Stop Streaming' : 'Start Streaming'}
            </button>
            <div id="status">${statusMessage ?? '...'}</div>
            ${error ? html`<div class="error">${error}</div>` : ''}
        `;
    }
}
customElements.define('stream-controls', StreamControls);


// ======================================================================== //
// Component: Card List (card-list)                                       //
// Uses StoreController for detections and selectedId                     //
// ======================================================================== //
class CardList extends LitElement {
    static styles = css` /* ... (Styles same as previous example) ... */
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

    // --- Store Controllers ---
    #detectionsController = new StoreController(this, $detections);
    #selectedIdController = new StoreController(this, $selectedId);

    #dispatchSelectionEvent(detectionId) {
         this.dispatchEvent(new CustomEvent('card-selected', {
             detail: { detectionId }, bubbles: true, composed: true }));
    }

    render() {
        const detections = this.#detectionsController.value ?? [];
        const selectedId = this.#selectedIdController.value;
        const sortedDetections = [...detections].sort((a, b) => (a?.id ?? 0) - (b?.id ?? 0));

        return html`
            ${sortedDetections.length === 0
                ? html`<p>No cards detected</p>`
                : sortedDetections.map(det => {
                      const bestMatch = det.matches?.[0];
                      const isSelected = det.id === selectedId;
                      return html`
                          <div
                              class="card-list-item ${isSelected ? 'selected' : ''}"
                              @click=${() => this.#dispatchSelectionEvent(det.id)}>
                              <div class="img-container">
                                  <img src=${'data:image/jpeg;base64,' + det.img} alt="Crop" loading="lazy">
                                  ${bestMatch?.img_uri
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
// Uses StoreController for selectedCardDetails                           //
// ======================================================================== //
class CardInfo extends LitElement {
    static styles = css` /* ... (Styles same as previous example) ... */
         :host { display: block; padding: 15px; background-color: var(--background-light, #3e3e3e); border-radius: var(--border-radius, 5px); margin: 0 10px 10px 10px; min-height: 100px; }
        h3 { margin: 0 0 5px 0; font-size: 1.1em; color: var(--text-primary, white); }
        p { margin: 5px 0; font-size: 0.9em; color: var(--text-secondary, #ccc); }
        img { width: 100%; height: auto; border-radius: 3px; margin-top: 10px; display: block; background-color: #555; }
        .placeholder { color: var(--text-secondary, #ccc); font-style: italic; }
    `;

    // --- Store Controller ---
    #detailsController = new StoreController(this, $selectedCardDetails);

    render() {
        const details = this.#detailsController.value; // Get current value
        if (!details) {
            return html`<p class="placeholder">Click a card to see details.</p>`;
        }

        return html`
            <h3>${details.name ?? 'Unknown Card'}</h3>
            <p>Set: ${details.set_name ?? 'Unknown'} (${details.set_code ?? 'N/A'})</p>
            ${details.img_uri
                ? html`<img src=${details.img_uri} alt=${details.name ?? 'Card image'} loading="lazy">`
                : html`<p class="placeholder">No image available</p>`}
        `;
    }
}
customElements.define('card-info', CardInfo);


// ======================================================================== //
// Component: Sidebar Container (side-bar)                                //
// Unchanged - simply provides slots                                      //
// ======================================================================== //
class SideBar extends LitElement {
    static styles = css` /* ... (Styles same as previous example) ... */
        :host { display: flex; flex-direction: column; width: var(--sidebar-width, 300px); background-color: var(--background-medium, #2e2e2e); padding-top: 10px; overflow: hidden; flex-shrink: 0; height: 100%; max-height: 100vh; }
        ::slotted(stream-controls) { flex-shrink: 0; }
        ::slotted(card-list) { flex-grow: 1; min-height: 50px; overflow-y: auto; }
        ::slotted(card-info) { flex-shrink: 0; }
         @media (max-width: 768px) { :host { width: 100%; height: auto; max-height: 50vh; flex: 1; padding-top: 0; } }
     `;
     render() { return html`<slot></slot>`; }
}
customElements.define('side-bar', SideBar);


// ======================================================================== //
// Component: Main Application (card-detector-app)                        //
// Orchestrates actions, updates stores, manages WebSocket & MediaStream. //
// ======================================================================== //
class CardDetectorApp extends LitElement {
    static styles = css` :host { /* Styles applied globally */ } `;

    // --- State Properties (Managed by LitElement - for non-store state) ---
    static properties = {
        _currentStream: { state: true }, // Keep stream state internal to Lit component
    };

    // --- Private Class Fields for Internal Refs & Logic ---
    #ws = null;
    #reconnectTimeout = null;
    #sendInterval = null;
    #videoElement = null; // Still need ref from child

    constructor() {
        super();
        this._currentStream = null; // Initialize Lit state
        console.log("CardDetectorApp constructed (Nano Stores version)");
    }

    connectedCallback() {
        super.connectedCallback();
        this.#connectWebSocket();
        (async () => {
            await this.#populateDevices();
            this.#attemptAutoStart();
        })();
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.#stopStreamingInternal();
        this.#ws?.close(1000, "Component disconnected");
        clearTimeout(this.#reconnectTimeout);
        clearInterval(this.#sendInterval);
        this.#ws = null;
    }

    // --- WebSocket Handling -> Updates Stores ---
    #connectWebSocket = () => {
        if (this.#ws?.readyState === WebSocket.OPEN || this.#ws?.readyState === WebSocket.CONNECTING) return;
        clearTimeout(this.#reconnectTimeout);
        $appStatus.setKey('statusMessage', 'Connecting to server...');
        $appStatus.setKey('error', null);

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/detect`;

        try {
            this.#ws = new WebSocket(wsUrl);
            this.#ws.onopen = this.#handleWsOpen;
            this.#ws.onmessage = this.#handleWsMessage;
            this.#ws.onerror = this.#handleWsError;
            this.#ws.onclose = this.#handleWsClose;
        } catch (e) {
            console.error("WebSocket constructor failed:", e);
            $appStatus.setKey('statusMessage', 'WebSocket init error.');
            $appStatus.setKey('error', `WebSocket URL invalid? ${e.message}`);
        }
    }

    #handleWsOpen = () => {
        console.log('WebSocket connected');
        $appStatus.setKey('statusMessage', 'Connected to server.');
        $appStatus.setKey('error', null);
        // Restart sending if needed
        if ($appStatus.get().isStreaming && !this.#sendInterval && this.#videoElement) {
            this.#startSendingFrames();
        }
    };

    #handleWsMessage = (event) => {
         try {
            const data = JSON.parse(event.data);
            const newDetections = data.detections ?? [];
            $detections.set(newDetections); // Update detections store

            // Handle selected card details implicitly via selection handler
            // Or check if selected still exists here
            const currentSelectedId = $selectedId.get();
            if (currentSelectedId !== null) {
                const currentSelected = newDetections.find(d => d.id === currentSelectedId);
                if (!currentSelected) {
                    $selectedId.set(null); // Deselect if gone
                    $selectedCardDetails.set(null);
                }
                // If it *still* exists, its details will be updated if selection handler runs again
                // or if we explicitly update details here
                // else { $selectedCardDetails.set(currentSelected?.matches?.[0] ?? null); }
            }
         } catch (e) {
             console.error("WS Message Parse Error:", e);
             $appStatus.setKey('statusMessage', 'Error processing server message.');
         }
    };

    #handleWsError = (error) => {
         console.error('WebSocket error:', error);
         $appStatus.setKey('statusMessage', 'WebSocket error.');
         $appStatus.setKey('error', 'Connection error.');
    };

    #handleWsClose = (event) => {
        console.log(`WebSocket closed. Code: ${event.code}`);
        clearInterval(this.#sendInterval);
        this.#sendInterval = null;
        const wasConnected = !!this.#ws;
        this.#ws = null;
        const currentStatus = $appStatus.get(); // Get current state before modifying

        if (event.code !== 1000 && event.code !== 1005 && wasConnected) {
             $appStatus.setKey('statusMessage', 'Disconnected. Reconnecting...');
             $appStatus.setKey('error', null); // Clear error for reconnect attempt
             this.#reconnectTimeout = setTimeout(this.#connectWebSocket, 5000);
        } else {
             $appStatus.setKey('statusMessage', 'Disconnected from server.');
        }
        // Always update streaming status if WS closes
        if (currentStatus.isStreaming) {
            $appStatus.setKey('isStreaming', false);
        }
    };


    // --- Video Stream Handling -> Updates Stores & Internal State ---
    #startStream = async (deviceId) => {
        console.log(`Attempting stream: ${deviceId ?? 'default'}`);
        this.#stopMediaStreamTracks();
        this._currentStream = null; // Clear Lit state
        $appStatus.setKey('isStreaming', false); // Ensure false until playing

        const constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } } };
        if (deviceId) constraints.video.deviceId = { exact: deviceId };

        try {
            $appStatus.setKey('statusMessage', 'Requesting camera...');
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this._currentStream = stream; // Set internal Lit state
            const actualDeviceId = stream.getVideoTracks()[0]?.getSettings()?.deviceId;
            $currentDeviceId.set(actualDeviceId ?? deviceId ?? $videoDevices.get()[0]?.deviceId);
            $appStatus.setKey('error', null);
            $appStatus.setKey('permissionGranted', true);
            console.log("getUserMedia success.");

            if (this.#videoElement) {
                 await new Promise(resolve => this.#videoElement.addEventListener('playing', resolve, { once: true }));
                 console.log("Stream playing, starting frames.");
                 $appStatus.setKey('statusMessage', 'Streaming to server...');
                 $appStatus.setKey('isStreaming', true); // Now set streaming true
                 this.#startSendingFrames(); // Start sending only when playing
            } else { console.warn("Video element ref missing for 'playing' listener."); }
            return true;

        } catch (error) {
            console.error('Camera Error:', error);
            let errorMsg = 'Error accessing camera.';
            let permission = null; // null = unknown
            if (error.name === 'NotAllowedError') { errorMsg = 'Camera access denied.'; permission = false; }
            else if (error.name === 'NotFoundError') { errorMsg = 'No camera found.'; permission = null; } // Can't grant if none found
            else if (error.name === 'NotReadableError') { errorMsg = 'Camera busy/unreadable.'; permission = true; } // Assume granted if readable error

            $appStatus.setKey('statusMessage', errorMsg);
            $appStatus.setKey('error', errorMsg);
            $appStatus.setKey('isStreaming', false);
            if (permission !== null) $appStatus.setKey('permissionGranted', permission);
            this._currentStream = null; // Clear Lit state
            return false;
        }
    }

    #populateDevices = async () => {
        console.log("Populating devices...");
        let devices = [];
        let permission = null; // Track permission status
        try {
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
            permission = true; // Granted if above succeeds
            tempStream.getTracks().forEach(track => track.stop());
        } catch (e) {
             if (e.name === 'NotAllowedError') { permission = false; }
             console.warn("Pre-enumeration getUserMedia failed:", e.name);
        }

        // Update permission state if determined
        if(permission !== null) $appStatus.setKey('permissionGranted', permission);

        if (permission === false) {
            $appStatus.setKey('statusMessage', "Camera access denied.");
            $appStatus.setKey('error', "Permission needed.");
            $videoDevices.set([]);
            $appStatus.setKey('devicesAvailable', false);
            return;
        }

        try {
            const enumeratedDevices = await navigator.mediaDevices.enumerateDevices();
            devices = enumeratedDevices.filter(device => device.kind === 'videoinput');
            $videoDevices.set(devices); // Update store
            $appStatus.setKey('devicesAvailable', devices.length > 0);
            console.log(`Found ${devices.length} video devices.`);
            if ($currentDeviceId.get() === null && devices.length > 0) {
                $currentDeviceId.set(devices[0].deviceId); // Set default in store
            }
             if (devices.length === 0) {
                 $appStatus.setKey('statusMessage', "No video devices found.");
                 $appStatus.setKey('error', "Connect a camera.");
             } else if (!$appStatus.get().error && permission === true) {
                 // If devices found and no other error, clear potential old errors
                 $appStatus.setKey('statusMessage', "Ready.");
                 $appStatus.setKey('error', null);
             }
        } catch (error) {
            console.error("Error enumerating devices:", error);
            $appStatus.setKey('statusMessage', "Error listing cameras.");
            $appStatus.setKey('error', "Could not get camera list.");
            $videoDevices.set([]);
            $appStatus.setKey('devicesAvailable', false);
        }
    }

    #startSendingFrames = () => {
        if (this.#sendInterval || !this.#videoElement || this.#videoElement.readyState < 3) return;
        if (this.#ws?.readyState !== WebSocket.OPEN) return;

        const canvas = document.createElement('canvas');
        canvas.width = this.#videoElement.videoWidth || 640;
        canvas.height = this.#videoElement.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        const frameRate = 10;

        console.log(`Starting frame sending (${frameRate} FPS).`);
        this.#sendInterval = setInterval(() => {
            if (this.#ws?.readyState === WebSocket.OPEN && this.#videoElement?.readyState >= 3) {
                try {
                    ctx.drawImage(this.#videoElement, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob(blob => {
                        if (blob?.size > 0 && this.#ws?.readyState === WebSocket.OPEN) {
                            this.#ws.send(blob);
                        }
                    }, 'image/jpeg', 0.5);
                } catch (e) { console.error("Canvas draw Error:", e); }
            }
        }, 1000 / frameRate);
    }

    #stopStreamingInternal = (updateStatus = true) => {
         console.log("Stopping streaming internals...");
         clearInterval(this.#sendInterval);
         this.#sendInterval = null;
         this.#stopMediaStreamTracks();
         this._currentStream = null; // Clear Lit state

         if(updateStatus) {
             $appStatus.setKey('isStreaming', false);
             $appStatus.setKey('statusMessage', 'Streaming stopped.');
             $detections.set([]);
             $selectedId.set(null);
             $selectedCardDetails.set(null);
         }
    }

    #stopMediaStreamTracks = () => {
        // Access Lit state property directly
        this._currentStream?.getTracks().forEach(track => track.stop());
    }


    #attemptAutoStart = async () => {
        console.log("Attempting auto-start...");
        // Check if we have devices and permission first
        const status = $appStatus.get();
        if (!status.devicesAvailable || status.permissionGranted === false) {
             console.log("Skipping auto-start due to no devices or permission denied.");
             // Status message should already reflect the issue from populateDevices
             return;
        }
        $appStatus.setKey('statusMessage', 'Attempting auto-start...');
        await this.#startStream($currentDeviceId.get()); // Use current device from store
    }

    // --- Event Handlers from Child Components -> Update Stores/Trigger Actions ---
    #handleVideoElementReady = (event) => {
        this.#videoElement = event.detail.videoElement;
        // If stream obtained before element ready, start sending now
        if (this._currentStream && !this.#sendInterval && $appStatus.get().isStreaming) {
            this.#startSendingFrames();
        }
    }

    #handleSelection = (event) => {
        const newId = event.detail.detectionId;
        $selectedId.set(newId); // Update selected ID store
        // Find details and update details store
        const selectedDetection = $detections.get().find(d => d.id === newId);
        $selectedCardDetails.set(selectedDetection?.matches?.[0] ?? null);
    }

    #handleDeviceChange = async (event) => {
        const newDeviceId = event.detail.deviceId;
        $appStatus.setKey('statusMessage', `Switching camera...`);
        this.#stopStreamingInternal(false); // Stop internals only
        await this.#startStream(newDeviceId); // Will update stores
    }

    #handleStartStream = async () => {
        $appStatus.setKey('statusMessage', 'Requesting camera...');
        $appStatus.setKey('error', null);
        await this.#startStream($currentDeviceId.get()); // Will update stores
    }

    #handleStopStream = () => {
        this.#stopStreamingInternal(); // Full stop, updates stores
    }


    // --- Render Method ---
    render() {
        // Renders layout. Child components get state via StoreController.
        // Passes non-store state (stream) down as prop.
        return html`
            <video-display
                .stream=${this._currentStream}
                @video-element-ready=${this.#handleVideoElementReady}
                @detection-selected=${this.#handleSelection}
            ></video-display>

            <side-bar>
                <stream-controls
                    @device-change=${this.#handleDeviceChange}
                    @start-stream=${this.#handleStartStream}
                    @stop-stream=${this.#handleStopStream}
                ></stream-controls>

                <card-list
                    @card-selected=${this.#handleSelection}
                ></card-list>

                <card-info></card-info> </side-bar>
        `;
    }
}
customElements.define('card-detector-app', CardDetectorApp);
