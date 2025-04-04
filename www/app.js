// Import LitElement base class, html, and css tag functions directly from CDN
import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@3/core/lit-core.min.js';

// ======================================================================== //
// Component: Video Display (video-display)                               //
// Handles video element, SVG overlay, and drawing detections.            //
// ======================================================================== //
class VideoDisplay extends LitElement {
    static styles = css`
        :host {
            display: block; /* Ensure it takes up space */
            position: relative;
            background-color: black;
            overflow: hidden; /* Clip contents */
            flex: 1; /* Grow to fill available space in flex container */
            min-width: 0; /* Allow shrinking in flex layouts */
            min-height: 200px; /* Minimum sensible height */
        }

        video, svg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: block; /* Remove extra space */
        }

        svg {
            pointer-events: none; /* Default, polygons will override */
            z-index: 1; /* Ensure SVG is on top */
        }

        /* Responsive adjustments if needed, though flex handles much of it */
         @media (max-width: 768px) {
             :host {
                height: 50vh; /* Specific height on mobile */
                flex: 0 1 auto; /* Don't grow, shrink if needed, base size auto */
                min-height: 250px;
             }
         }
    `;

    // --- Properties passed down from parent ---
    static properties = {
        stream: { type: Object }, // MediaStream object
        detections: { type: Array }, // Array of detection objects
        selectedId: { type: Number, nullable: true }, // ID of the selected detection
    };

    // --- Internal state ---
    svgInstance = null; // Instance of SVG.js library
    videoElement = null; // Reference to the video element

    constructor() {
        super();
        this.stream = null;
        this.detections = [];
        this.selectedId = null;
        this._resizeObserver = null; // For observing container size changes
    }

    connectedCallback() {
        super.connectedCallback();
        // Set up resize observer to handle element resizing
        this._resizeObserver = new ResizeObserver(() => this.updateOverlaySize());
        this._resizeObserver.observe(this); // Observe the component itself
        // console.log("video-display connected");
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        if (this.svgInstance) {
            try {
                 this.svgInstance.remove(); // Clean up SVG.js instance
            } catch (e) { console.warn("Error removing SVG.js instance:", e)}
            this.svgInstance = null;
        }
        // console.log("video-display disconnected");
    }

    firstUpdated() {
        // Get references to internal elements after first render
        this.videoElement = this.shadowRoot.getElementById('video');
        const svgElement = this.shadowRoot.getElementById('overlay');

        // Initialize SVG.js
        if (svgElement && typeof SVG === 'function') { // Check if SVG.js is loaded
             try {
                 this.svgInstance = SVG(svgElement); // Attach to the SVG element in shadow DOM
                 this.updateOverlaySize(); // Set initial size/viewbox
                 // console.log("SVG.js initialized in video-display");
             } catch (e) {
                 console.error("Failed to initialize SVG.js:", e);
             }
        } else if (typeof SVG !== 'function') {
            console.error("SVG.js library not loaded.");
        } else {
            console.error("SVG overlay element not found in shadow DOM.");
        }

        // Emit event to notify parent that the video element is ready
        if (this.videoElement) {
            this.dispatchEvent(new CustomEvent('video-element-ready', {
                detail: { videoElement: this.videoElement },
                bubbles: true, // Allow event to bubble up
                composed: true // Allow event to cross shadow DOM boundary
            }));
        } else {
            console.error("Video element not found in shadow DOM during firstUpdated.");
        }

        // Listen for video metadata loading to update overlay size
        this.videoElement.addEventListener('loadedmetadata', () => this.updateOverlaySize());
        // console.log("video-display firstUpdated complete");
    }

    // Called whenever properties change
    updated(changedProperties) {
        // Update video srcObject when stream property changes
        if (changedProperties.has('stream') && this.videoElement) {
            // console.log("Stream property changed, updating srcObject");
            this.videoElement.srcObject = this.stream;
        }
        // Redraw detections if detections or selectedId change
        if ((changedProperties.has('detections') || changedProperties.has('selectedId')) && this.svgInstance) {
             // console.log("Detections or selectedId changed, redrawing SVG");
             this.drawDetections();
        }
         // Re-check overlay size if stream changes (might change video dimensions)
         if (changedProperties.has('stream')) {
            // Wait a tick for video dimensions potentially updating
            requestAnimationFrame(() => this.updateOverlaySize());
         }
    }

    updateOverlaySize() {
        if (!this.videoElement || !this.svgInstance || !this.isConnected) {
             // console.warn("Cannot update overlay size: element or SVG instance missing, or disconnected.");
             return; // Don't run if elements aren't ready or component is disconnected
        }
        const { videoWidth, videoHeight } = this.videoElement; // Use intrinsic video size
        const { width, height } = this.getBoundingClientRect(); // Use display size for scaling reference

        if (videoWidth > 0 && videoHeight > 0) {
            // Set the viewbox to match the video's intrinsic resolution
            try {
                 this.svgInstance.viewbox(0, 0, videoWidth, videoHeight);
                 // console.log(`Overlay viewbox set to ${videoWidth}x${videoHeight}`);
            } catch (e) {
                 console.error("Error setting viewbox:", e);
            }
        } else if (width > 0 && height > 0) {
             // Fallback using component's bounding rect if video dimensions aren't ready
             // This might be less accurate for drawing coordinates if aspect ratios differ.
             // console.warn("Video dimensions not ready, using component bounds for viewbox (may be inaccurate).");
             // this.svgInstance.viewbox(0, 0, width, height); // Alternative fallback
        }
    }

    drawDetections() {
        if (!this.svgInstance) return;

        try {
             this.svgInstance.clear(); // Clear previous drawings
        } catch (e) {
             console.error("Error clearing SVG:", e); return;
        }


        const detectionsToDraw = this.detections || [];

        detectionsToDraw.forEach(det => {
            if (!det || !det.points || !Array.isArray(det.points)) {
                console.warn("Skipping invalid detection object:", det);
                return;
            }

            const isSelected = det.id === this.selectedId;
            const strokeColor = isSelected ? 'var(--selection-color, yellow)' : (det.color || 'var(--accent-color, lime)');
            const strokeWidth = isSelected ? 4 : 2;

            const validPoints = det.points.filter(p => Array.isArray(p) && p.length === 2 && !isNaN(p[0]) && !isNaN(p[1]));
            if (validPoints.length < 3) return; // Need 3+ points
            const pointsString = validPoints.map(p => p.join(',')).join(' ');

            try {
                const polygon = this.svgInstance.polygon(pointsString)
                    .fill('none')
                    .stroke({ color: strokeColor, width: strokeWidth })
                    .attr('pointer-events', 'auto') // Enable clicking
                    .on('click', (event) => {
                        event.stopPropagation();
                        // Dispatch event upwards to notify parent of selection
                        this.dispatchEvent(new CustomEvent('detection-selected', {
                            detail: { detectionId: det.id },
                            bubbles: true,
                            composed: true
                        }));
                    });

                const bestMatch = det.matches && det.matches.length > 0 ? det.matches[0] : null;
                if (bestMatch && bestMatch.name) {
                    const topPoint = validPoints.reduce((a, b) => a[1] < b[1] ? a : b);
                    if (topPoint) {
                        this.svgInstance.text(bestMatch.name)
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
// Handles camera selection, start/stop button, and status display.       //
// ======================================================================== //
class StreamControls extends LitElement {
    static styles = css`
        :host {
            display: block;
            padding: 10px;
            margin-bottom: 15px; /* Space below controls */
        }
        select, button {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border-radius: var(--border-radius, 5px);
            font-family: var(--font-family);
            font-size: 16px;
            cursor: pointer;
        }
        select {
            background-color: var(--background-light, #333);
            color: var(--text-primary, #fff);
            border: 2px solid var(--accent-color, #00cc00);
        }
        button {
            background-color: transparent;
            color: var(--accent-color, #00cc00);
            border: 2px solid var(--accent-color, #00cc00);
            transition: background-color 0.3s, color 0.3s;
        }
        button:hover:not(:disabled) {
            background-color: var(--accent-color, #00cc00);
            color: var(--background-dark, #1e1e1e);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            border-color: var(--text-secondary, #ccc);
            color: var(--text-secondary, #ccc);
        }
        #status {
            margin-top: 10px;
            font-size: 14px;
            color: var(--text-secondary, #ccc);
            min-height: 1.2em; /* Prevent layout shift */
        }
        .error {
            color: var(--error-color, red);
            margin-top: 5px;
            font-size: 14px;
        }
    `;

    static properties = {
        videoDevices: { type: Array },
        currentDeviceId: { type: String, nullable: true },
        isStreaming: { type: Boolean },
        statusMessage: { type: String },
        error: { type: String, nullable: true },
        canStart: { type: Boolean } // Whether start button should be enabled (permission etc)
    };

    constructor() {
        super();
        this.videoDevices = [];
        this.currentDeviceId = null;
        this.isStreaming = false;
        this.statusMessage = "Initializing...";
        this.error = null;
        this.canStart = false; // Start disabled until devices loaded/checked
    }

    _handleDeviceChange(event) {
        this.dispatchEvent(new CustomEvent('device-change', {
            detail: { deviceId: event.target.value },
            bubbles: true, composed: true
        }));
    }

    _handleStartStopClick() {
        const eventName = this.isStreaming ? 'stop-stream' : 'start-stream';
        this.dispatchEvent(new CustomEvent(eventName, { bubbles: true, composed: true }));
    }

    render() {
        return html`
            <select id="select" @change=${this._handleDeviceChange} ?disabled=${this.videoDevices.length === 0 || this.isStreaming}>
                ${this.videoDevices.length === 0
                    ? html`<option>No cameras found</option>`
                    : this.videoDevices.map(device => html`
                        <option value=${device.deviceId} ?selected=${device.deviceId === this.currentDeviceId}>
                            ${device.label || `Camera ${this.videoDevices.indexOf(device) + 1}`}
                        </option>
                    `)
                }
            </select>
            <button
                id="startCamera"
                @click=${this._handleStartStopClick}
                ?disabled=${!this.canStart}>
                ${this.isStreaming ? 'Stop Streaming' : 'Start Streaming'}
            </button>
            <div id="status">${this.statusMessage}</div>
            ${this.error ? html`<div class="error">${this.error}</div>` : ''}
        `;
    }
}
customElements.define('stream-controls', StreamControls);


// ======================================================================== //
// Component: Card List (card-list)                                       //
// Displays the list of detected cards.                                   //
// ======================================================================== //
class CardList extends LitElement {
    static styles = css`
        :host {
            display: block; /* Takes up block space */
            flex: 1; /* Grow to fill available vertical space */
            overflow-y: auto; /* Enable vertical scrolling */
            padding: 0 10px; /* Padding on the sides */
            margin-bottom: 10px; /* Space below list */
        }
        .card-list-item {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            cursor: pointer;
            padding: 5px;
            border: 2px solid transparent; /* Placeholder for selection border */
            border-radius: var(--border-radius, 4px);
            transition: background-color 0.2s, border-color 0.2s;
            background-color: var(--background-medium, #2e2e2e);
         }
         .card-list-item:hover {
            background-color: var(--background-light, #3a3a3a);
         }
         .card-list-item.selected {
             border-color: var(--selection-color, yellow);
             background-color: #444; /* Slightly different background when selected */
         }
         .img-container {
            position: relative;
            width: 50px;
            height: 70px;
            overflow: hidden;
            border-radius: 3px;
            flex-shrink: 0;
            background-color: #555; /* Placeholder bg */
         }
         img {
             display: block;
             width: 100%;
             height: auto;
         }
         .uri-img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
         }
         .info {
            margin-left: 10px;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            font-size: 14px;
         }
         .info strong {
             display: block; /* Put ID on its own line effectively */
             margin-bottom: 3px;
         }
         p {
             color: var(--text-secondary, #ccc);
             text-align: center;
             margin-top: 20px;
         }
    `;

    static properties = {
        detections: { type: Array },
        selectedId: { type: Number, nullable: true }
    };

    constructor() {
        super();
        this.detections = [];
        this.selectedId = null;
    }

    _handleItemClick(detectionId) {
        this.dispatchEvent(new CustomEvent('card-selected', {
            detail: { detectionId: detectionId },
            bubbles: true, composed: true
        }));
    }

    render() {
        const sortedDetections = [...this.detections].sort((a, b) => (a.id || 0) - (b.id || 0));

        return html`
            ${sortedDetections.length === 0
                ? html`<p>No cards detected</p>`
                : sortedDetections.map(det => {
                      const bestMatch = det.matches && det.matches.length > 0 ? det.matches[0] : null;
                      const isSelected = det.id === this.selectedId;
                      return html`
                          <div
                              class="card-list-item ${isSelected ? 'selected' : ''}"
                              @click=${() => this._handleItemClick(det.id)}>
                              <div class="img-container">
                                  <img src=${'data:image/jpeg;base64,' + det.img} alt="Detected Card Crop">
                                  ${bestMatch && bestMatch.img_uri
                                      ? html`<img class="uri-img" src=${bestMatch.img_uri} alt="Best Match Image" loading="lazy">`
                                      : ''}
                              </div>
                              <div class="info">
                                  <strong>ID: ${det.id}</strong>
                                  ${bestMatch ? bestMatch.name : 'Unknown'}
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
// Displays details of the selected card.                                 //
// ======================================================================== //
class CardInfo extends LitElement {
    static styles = css`
        :host {
            display: block;
            padding: 15px;
            background-color: var(--background-light, #3e3e3e);
            border-radius: var(--border-radius, 5px);
            margin: 0 10px 10px 10px; /* Add some margin */
            min-height: 100px; /* Ensure it takes some space */
        }
        h3 {
            margin: 0 0 5px 0;
            font-size: 1.1em;
            color: var(--text-primary, white);
        }
        p {
            margin: 5px 0;
            font-size: 0.9em;
            color: var(--text-secondary, #ccc);
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 3px;
            margin-top: 10px;
            display: block;
            background-color: #555; /* Placeholder */
        }
        .placeholder {
            color: var(--text-secondary, #ccc);
            font-style: italic;
        }
    `;

    static properties = {
        // Pass the whole details object for simplicity
        selectedCardDetails: { type: Object, nullable: true }
    };

    constructor() {
        super();
        this.selectedCardDetails = null;
    }

    render() {
        if (!this.selectedCardDetails) {
            return html`<p class="placeholder">Click a card to see details.</p>`;
        }

        const details = this.selectedCardDetails;
        // Use Scryfall image URI if available, otherwise could show detection img (but it's small)
        const imageUrl = details.img_uri; // Prefer Scryfall image

        return html`
            <h3>${details.name || 'Unknown Card'}</h3>
            <p>
                Set: ${details.set_name || 'Unknown'} (${details.set_code || 'N/A'})
            </p>
            ${imageUrl
                ? html`<img src=${imageUrl} alt=${details.name || 'Card image'} loading="lazy">`
                : html`<p class="placeholder">No image available</p>`}
            `;
    }
}
customElements.define('card-info', CardInfo);


// ======================================================================== //
// Component: Sidebar Container (side-bar)                                //
// Optional: Provides layout structure for sidebar items.                 //
// ======================================================================== //
class SideBar extends LitElement {
     static styles = css`
        :host {
            display: flex;
            flex-direction: column;
            width: var(--sidebar-width, 300px);
            background-color: var(--background-medium, #2e2e2e);
            padding-top: 10px; /* Add padding at the top */
            overflow: hidden; /* Prevent content spilling out */
            flex-shrink: 0; /* Prevent sidebar from shrinking */
            height: 100%; /* Default height */
            max-height: 100vh; /* Ensure it doesn't exceed viewport height */
        }

        /* Ensure children layout correctly */
        ::slotted(stream-controls) {
            flex-shrink: 0; /* Prevent controls from shrinking */
        }
        ::slotted(card-list) {
            flex-grow: 1; /* Allow list to take up remaining space */
            min-height: 50px; /* Ensure list has some minimum space */
            overflow-y: auto; /* Allow list itself to scroll */
        }
         ::slotted(card-info) {
             flex-shrink: 0; /* Prevent info box from shrinking */
         }

         @media (max-width: 768px) {
             :host {
                 width: 100%; /* Full width on mobile */
                 height: auto; /* Adjust height automatically */
                 max-height: 50vh; /* Limit height on mobile */
                 flex: 1; /* Allow it to grow if needed in column layout */
                 padding-top: 0;
             }
         }
     `;

     render() {
         // Use <slot> to allow parent component to project children into the sidebar
         return html`<slot></slot>`;
     }
}
customElements.define('side-bar', SideBar);


// ======================================================================== //
// Component: Main Application (card-detector-app)                        //
// Orchestrates components, handles WebSocket, stream management & state. //
// ======================================================================== //
class CardDetectorApp extends LitElement {
    static styles = css`
        :host {
            /* Already styled globally */
            /* display: flex; (set globally) */
            /* flex-direction: row; (set globally) */
        }
        /* No component-specific styles needed if layout handled by global styles + sidebar component */
    `;

    // --- State Management ---
    static properties = {
        // Data received from WebSocket or generated locally
        _detections: { state: true, type: Array },
        _selectedId: { state: true, type: Number, nullable: true },
        _selectedCardDetails: { state: true, type: Object, nullable: true },
        _videoDevices: { state: true, type: Array },
        _currentDeviceId: { state: true, type: String, nullable: true },

        // Status and Control Flow
        _isStreaming: { state: true, type: Boolean },
        _statusMessage: { state: true, type: String },
        _error: { state: true, type: String, nullable: true },
        _canStartManually: { state: true, type: Boolean }, // Controls button enabled state

        // Internal Handles and Objects
        _currentStream: { state: true, type: Object, nullable: true }, // Internal reference to the stream
        _videoElement: { state: true, type: Object, nullable: true } // Reference from video-display component
    };

    _ws = null; // WebSocket instance
    _reconnectTimeout = null;
    _sendInterval = null; // Interval ID for sending frames
    _isAutoStarting = true; // Flag during initial load

    constructor() {
        super();
        this._detections = [];
        this._selectedId = null;
        this._selectedCardDetails = null;
        this._videoDevices = [];
        this._currentDeviceId = null;
        this._isStreaming = false;
        this._statusMessage = "Initializing...";
        this._error = null;
        this._canStartManually = false; // Disabled initially
        this._currentStream = null;
        this._videoElement = null;

         console.log("CardDetectorApp constructed");
    }

    connectedCallback() {
        super.connectedCallback();
        console.log("CardDetectorApp connected");
        this.connectWebSocket();
        this.populateDevices().then(() => {
            this.attemptAutoStart();
        });
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        console.log("CardDetectorApp disconnected");
        this.stopStreaming(); // Stop stream if component is removed
        if (this._ws) {
            this._ws.close(1000, "Component disconnected"); // Close WebSocket cleanly
            this._ws = null;
        }
        clearTimeout(this._reconnectTimeout);
        clearInterval(this._sendInterval);
        this._sendInterval = null;
    }

    // --- WebSocket Handling ---
    connectWebSocket() {
        if (this._ws && (this._ws.readyState === WebSocket.CONNECTING || this._ws.readyState === WebSocket.OPEN)) {
            return;
        }
        clearTimeout(this._reconnectTimeout);

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/detect`;

        this._statusMessage = 'Connecting to server...';
        try {
            this._ws = new WebSocket(wsUrl);
        } catch (e) {
             console.error("WebSocket constructor failed:", e);
             this._statusMessage = "Error initializing WebSocket.";
             this._error = `WebSocket URL invalid? ${e.message}`;
             this._canStartManually = false; // Can't start if WS fails
             return;
        }


        this._ws.onopen = () => {
            console.log('WebSocket connection established');
            this._statusMessage = 'Connected to server.';
            this._error = null; // Clear previous errors
            // If streaming was intended, restart sending frames
            if (this._isStreaming && !this._sendInterval && this._videoElement) {
                this.startSendingFrames();
            }
        };

        this._ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const newDetections = data.detections || [];
                this._detections = newDetections; // Update detections state

                // Update selected card details if the selected card is still present
                if (this._selectedId !== null) {
                    const currentSelected = newDetections.find(d => d.id === this._selectedId);
                    if (currentSelected) {
                        this._selectedCardDetails = (currentSelected.matches && currentSelected.matches.length > 0) ? currentSelected.matches[0] : null;
                    } else {
                        // Selected card disappeared
                        this._selectedId = null;
                        this._selectedCardDetails = null;
                    }
                }
            } catch (e) {
                console.error("Error parsing WebSocket message:", e, event.data);
                this._statusMessage = 'Error processing server message.';
                // Avoid setting a persistent error here unless it's fatal
            }
        };

        this._ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this._statusMessage = 'WebSocket error occurred.';
            this._error = 'Connection to detection server failed.';
            // Might trigger close event anyway
        };

        this._ws.onclose = (event) => {
            console.log(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason}`);
             if (this._sendInterval) {
                clearInterval(this._sendInterval);
                this._sendInterval = null;
             }

            this._ws = null; // Clear instance ref

            if (event.code !== 1000 && event.code !== 1005) { // Avoid reconnect on normal/manual close
                this._statusMessage = 'Disconnected. Reconnecting...';
                this._reconnectTimeout = setTimeout(() => this.connectWebSocket(), 5000);
            } else if (this._isStreaming) { // If it closed normally but we thought we were streaming
                 this._statusMessage = 'Server disconnected.';
                 this._isStreaming = false; // Update state if closed unexpectedly
                 this._canStartManually = true; // Allow manual restart
            } else {
                 this._statusMessage = 'Disconnected from server.';
            }
        };
    }

    // --- Video Stream Handling ---
    async startStream(deviceId) {
        console.log(`Attempting to start stream for device: ${deviceId || 'default'}`);
        if (this._currentStream) {
            this._currentStream.getTracks().forEach(track => track.stop());
            this._currentStream = null; // Update state
        }

        const constraints = {
            video: {
                width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 }
            }
        };
        if (deviceId) {
            constraints.video.deviceId = { exact: deviceId };
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this._currentStream = stream; // Update reactive property -> passes to video-display
            const actualDeviceId = stream.getVideoTracks()[0]?.getSettings()?.deviceId;
            this._currentDeviceId = actualDeviceId || deviceId || this._videoDevices[0]?.deviceId;
             // Don't set isStreaming true until frames are actually sending
            this._error = null; // Clear previous errors
            console.log("getUserMedia successful.");

            // Wait for the video element to report it's playing before sending frames
            if (this._videoElement) {
                 this._videoElement.addEventListener('playing', () => {
                      console.log("Stream playing, starting frame sending.");
                      this.startSendingFrames();
                      this._isStreaming = true; // Now we are officially streaming
                      this._statusMessage = 'Streaming to server...';
                 }, { once: true });
            } else {
                console.warn("Video element reference not available yet to listen for 'playing' event.");
                // Frame sending will be attempted later when video element is ready, if needed
            }
            return true; // Indicate success

        } catch (error) {
            console.error('Error accessing camera:', error);
            let errorMsg = 'Error: Could not access camera.';
            if (error.name === 'NotAllowedError') errorMsg = 'Camera access denied. Please grant permission.';
            else if (error.name === 'NotFoundError') errorMsg = 'No camera found with specified criteria.';
            else if (error.name === 'NotReadableError') errorMsg = 'Camera is already in use or unavailable.';

            this._statusMessage = errorMsg;
            this._error = errorMsg;
            this._isStreaming = false;
            this._currentStream = null;
            this._canStartManually = false; // Can't start if permission denied
            return false; // Indicate failure
        }
    }

    async populateDevices() {
        console.log("Populating video devices...");
        try {
            // Ensure permissions are prompted if needed, required for enumerateDevices labels
            await navigator.mediaDevices.getUserMedia({ video: true });
             // Immediately stop the track obtained just for permission/listing
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
            tempStream.getTracks().forEach(track => track.stop());

        } catch (e) {
            console.warn("Error during pre-enumeration getUserMedia (may be expected if permission denied):", e.name);
             if (e.name === 'NotAllowedError') {
                this._statusMessage = "Camera access denied. Cannot list devices.";
                this._error = "Permission needed to list cameras.";
                this._canStartManually = false;
                return;
             }
             // Continue if it's another error, enumerateDevices might still work partially
        }

        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            this._videoDevices = devices.filter(device => device.kind === 'videoinput');
            console.log(`Found ${this._videoDevices.length} video devices.`);
            // Set default device ID if not already set
            if (!this._currentDeviceId && this._videoDevices.length > 0) {
                this._currentDeviceId = this._videoDevices[0].deviceId;
            }
             this._canStartManually = this._videoDevices.length > 0; // Enable start only if devices found
        } catch (error) {
            console.error("Error enumerating devices:", error);
            this._statusMessage = "Error listing cameras.";
            this._error = "Could not get camera list.";
            this._canStartManually = false;
        }
    }

    startSendingFrames() {
        if (this._sendInterval) return; // Already sending
        if (!this._videoElement || this._videoElement.readyState < this._videoElement.HAVE_METADATA) {
            console.warn("Video element not ready for sending frames.");
            return;
        }
         if (!this._ws || this._ws.readyState !== WebSocket.OPEN) {
             console.warn("WebSocket not open. Cannot start sending frames.");
             return;
         }


        const canvas = document.createElement('canvas');
        canvas.width = this._videoElement.videoWidth || 640;
        canvas.height = this._videoElement.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        const frameRate = 10; // Target FPS

        console.log(`Starting frame sending interval (${frameRate} FPS). Canvas: ${canvas.width}x${canvas.height}`);

        this._sendInterval = setInterval(() => {
            if (this._ws && this._ws.readyState === WebSocket.OPEN && this._videoElement.readyState >= this._videoElement.HAVE_CURRENT_DATA) {
                 try {
                    ctx.drawImage(this._videoElement, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob(blob => {
                        if (blob && blob.size > 0 && this._ws && this._ws.readyState === WebSocket.OPEN) {
                            this._ws.send(blob);
                        } else if (blob && blob.size === 0) {
                             console.warn("Generated blob size is 0, not sending.");
                        }
                    }, 'image/jpeg', 0.5);
                 } catch (drawError) {
                     console.error("Error drawing video to canvas:", drawError);
                 }
            } else if (!this._ws || this._ws.readyState !== WebSocket.OPEN) {
                 console.warn("WebSocket closed or closing, stopping frame send interval.");
                 clearInterval(this._sendInterval);
                 this._sendInterval = null;
                 // Update state? Maybe rely on ws.onclose handler
            }
        }, 1000 / frameRate);

        // Ensure isStreaming is true (might be redundant if set via playing event)
        this._isStreaming = true;
    }

    stopStreaming() {
        console.log("Stopping streaming...");
        if (this._sendInterval) {
            clearInterval(this._sendInterval);
            this._sendInterval = null;
        }
        if (this._currentStream) {
            this._currentStream.getTracks().forEach(track => track.stop());
        }
        this._currentStream = null; // Update state to remove stream from video-display
        this._isStreaming = false;
        this._statusMessage = 'Streaming stopped.';
        this._detections = []; // Clear detections
        this._selectedId = null;
        this._selectedCardDetails = null;
        this._canStartManually = this._videoDevices.length > 0; // Re-enable start if possible
         console.log("Streaming stopped successfully.");
    }

    async attemptAutoStart() {
        console.log("Attempting auto-start...");
        this._isAutoStarting = true;
        this._statusMessage = 'Attempting auto-start...';

        const success = await this.startStream(this._currentDeviceId); // Use default/first device

        if (success) {
            console.log("Auto-start stream acquired.");
            // Frame sending and status updates are handled by event listener in startStream
        } else {
            console.warn('Auto-start failed. User interaction likely required.');
            // Error messages handled within startStream
        }
        this._isAutoStarting = false;
        // canStartManually state should be set correctly by startStream/populateDevices
    }

    // --- Event Handlers from Child Components ---
    _handleVideoElementReady(event) {
        console.log("Received video-element-ready event");
        this._videoElement = event.detail.videoElement;
        // If we were already trying to stream, start sending frames now
        if (this._isStreaming && !this._sendInterval && this._currentStream) {
             console.log("Video element ready, attempting to start frame sending now.");
             this.startSendingFrames();
        }
    }

    _handleDetectionSelected(event) {
        const newId = event.detail.detectionId;
        console.log(`Detection selected event: ID ${newId}`);
        this._selectedId = newId;
        // Find details for the selected card
        const selectedDetection = this._detections.find(d => d.id === newId);
        this._selectedCardDetails = (selectedDetection?.matches && selectedDetection.matches.length > 0)
            ? selectedDetection.matches[0]
            : null;
    }

    _handleCardSelected(event) {
         // Same logic as detection selected from the overlay
         this._handleDetectionSelected(event);
    }

    async _handleDeviceChange(event) {
        const newDeviceId = event.detail.deviceId;
        console.log(`Device change requested: ${newDeviceId}`);
        this._statusMessage = `Switching camera...`;
        // Stop sending frames before changing stream
        if (this._sendInterval) {
            clearInterval(this._sendInterval);
            this._sendInterval = null;
        }
        this._isStreaming = false; // Mark as not streaming during switch
        const success = await this.startStream(newDeviceId);
        if (!success) {
            this._statusMessage = 'Failed to switch camera.';
             this._canStartManually = this._videoDevices.length > 0; // Re-enable button if possible
        }
        // Status messages/frame sending handled by startStream's playing listener
    }

    async _handleStartStream() {
        console.log("Start stream requested");
        this._isAutoStarting = false; // Ensure auto-start flag is off
        this._statusMessage = 'Requesting camera access...';
        this._error = null; // Clear previous errors
        const success = await this.startStream(this._currentDeviceId);
        if (success) {
            // Status message/streaming state handled by 'playing' listener in startStream
        } else {
             // Error state handled within startStream
        }
    }

    _handleStopStream() {
        console.log("Stop stream requested");
        this.stopStreaming();
    }


    // --- Render Method ---
    render() {
        // Pass state down to child components as properties
        // Listen for events bubbling up from child components
        return html`
            <video-display
                .stream=${this._currentStream}
                .detections=${this._detections}
                .selectedId=${this._selectedId}
                @video-element-ready=${this._handleVideoElementReady}
                @detection-selected=${this._handleDetectionSelected}
            ></video-display>

            <side-bar>
                <stream-controls
                    .videoDevices=${this._videoDevices}
                    .currentDeviceId=${this._currentDeviceId}
                    .isStreaming=${this._isStreaming}
                    .statusMessage=${this._statusMessage}
                    .error=${this._error}
                    .canStart=${this._canStartManually && !this._isAutoStarting}
                    @device-change=${this._handleDeviceChange}
                    @start-stream=${this._handleStartStream}
                    @stop-stream=${this._handleStopStream}
                ></stream-controls>

                <card-list
                    .detections=${this._detections}
                    .selectedId=${this._selectedId}
                    @card-selected=${this._handleCardSelected}
                ></card-list>

                <card-info
                    .selectedCardDetails=${this._selectedCardDetails}
                ></card-info>
            </side-bar>
        `;
    }
}
customElements.define('card-detector-app', CardDetectorApp);
