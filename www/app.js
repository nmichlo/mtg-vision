import { html, render } from 'https://cdn.jsdelivr.net/npm/lit-html@3/lit-html.js';
// If using npm: import {html, render} from 'lit-html';
import { unsafeHTML } from 'https://cdn.jsdelivr.net/npm/lit-html@3/directives/unsafe-html.js';
// If using npm: import {unsafeHTML} from 'lit-html/directives/unsafe-html.js';


// ======================================================================== //
// Application State                                                        //
// ======================================================================== //

let state = {
    currentStream: null,
    ws: null,
    selectedId: null,
    selectedCardDetails: null, // Store details of the selected card for rendering
    reconnectTimeout: null,
    svgInstance: null, // SVG.js instance
    isStreaming: false, // Track streaming state
    sendInterval: null, // Store interval for sending frames
    statusMessage: 'Connecting to server...',
    videoDevices: [],
    currentDeviceId: null,
    detections: [],
    isAutoStarting: true, // Flag to manage initial auto-start UI
    canStartManually: true, // Flag to enable/disable start button based on permissions etc.
    error: null, // Store potential errors
};

// ======================================================================== //
// WebSocket Handling                                                       //
// ======================================================================== //

function connectWebSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.CONNECTING || state.ws.readyState === WebSocket.OPEN)) {
        console.log("WebSocket already connecting or open.");
        return;
    }
    clearTimeout(state.reconnectTimeout); // Clear any pending reconnect timeout

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/detect`;

    console.log(`Attempting to connect WebSocket to ${wsUrl}`);
    state.ws = new WebSocket(wsUrl);
    updateState({ statusMessage: 'Connecting to server...' });


    state.ws.onopen = () => {
        console.log('WebSocket connection established');
        updateState({ statusMessage: 'Connected to server.' });
        // If streaming was intended but stopped due to disconnect, restart sending
        if (state.isStreaming && !state.sendInterval) {
            const videoElement = document.getElementById('video');
            if (videoElement && videoElement.srcObject) {
                startSendingFrames(videoElement);
            }
        }
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            // console.log('Detections received:', data.detections); // Less verbose log
            const newDetections = data.detections || [];

            let newSelectedCardDetails = state.selectedCardDetails;
            // If a card was selected, check if it still exists
            if (state.selectedId !== null) {
                const stillExists = newDetections.find(det => det.id === state.selectedId);
                if (stillExists) {
                    // Update details in case they changed (though unlikely for static data)
                    newSelectedCardDetails = stillExists.matches[0] || null;
                } else {
                    // Selected card disappeared
                    console.log(`Selected card ${state.selectedId} no longer detected.`);
                    state.selectedId = null; // Deselect
                    newSelectedCardDetails = null;
                }
            }

            // Update state triggers re-render which calls drawDetections
            updateState({
                detections: newDetections,
                selectedCardDetails: newSelectedCardDetails
            });

        } catch (e) {
            console.error("Error parsing WebSocket message:", e);
            console.error("Received data:", event.data);
            updateState({ statusMessage: 'Error processing server message.' });
        }
    };

    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateState({ statusMessage: 'WebSocket error occurred.', error: 'WebSocket connection failed.' });
        // Consider attempting reconnect here or relying on onclose
    };

    state.ws.onclose = (event) => {
        console.log(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason}`);
        // Clear send interval if connection closes
        if (state.sendInterval) {
            clearInterval(state.sendInterval);
            state.sendInterval = null;
            console.log("Stopped sending frames due to WebSocket close.");
        }

        let message = 'Disconnected from server.';
        // Attempt to reconnect unless closed cleanly or manually stopped
        if (event.code !== 1000 && event.code !== 1005) { // 1000 = Normal Closure, 1005 = No Status Rcvd (often indicates manual stop)
            message += ' Reconnecting in 5 seconds...';
            console.log("Attempting WebSocket reconnection...");
            state.reconnectTimeout = setTimeout(connectWebSocket, 5000);
        }
        updateState({ statusMessage: message, ws: null }); // Set ws to null to indicate closure
    };
}


// ======================================================================== //
// Video Stream Handling                                                    //
// ======================================================================== //

async function startStream(deviceId) {
    console.log(`Attempting to start stream for device: ${deviceId || 'default'}`);
    if (state.currentStream) {
        console.log("Stopping existing stream tracks.");
        state.currentStream.getTracks().forEach(track => track.stop());
        updateState({ currentStream: null }); // Update state immediately
    }
    // Ensure video element exists before proceeding
     const videoElement = document.getElementById('video');
     if (!videoElement) {
         console.error("Video element not found");
         updateState({statusMessage: "Error: Video element missing.", error: "Video element not found."});
         return null; // Return null or throw error
     }

    const constraints = {
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 15 } // Request a reasonable frame rate
        }
    };
    if (deviceId) {
        constraints.video.deviceId = { exact: deviceId };
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log("getUserMedia successful.");
        updateState({ currentStream: stream, currentDeviceId: deviceId || stream.getVideoTracks()[0].getSettings().deviceId });

        // The video element's srcObject is now handled by the template binding
        // Ensure overlay size is updated once video metadata loads
        videoElement.addEventListener('loadedmetadata', updateOverlaySize, { once: true });

        return stream; // Return the stream for chaining if needed
    } catch (error) {
        console.error('Error accessing camera:', error);
        let errorMsg = 'Error: Could not access camera.';
        if (error.name === 'NotAllowedError') {
            errorMsg = 'Camera access denied. Please grant permission.';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            errorMsg = 'No camera found.';
        } else if (error.name === 'NotReadableError') {
            errorMsg = 'Camera is already in use or unavailable.';
        }
        updateState({
            statusMessage: errorMsg,
            error: errorMsg,
            isStreaming: false, // Ensure streaming state is false
            canStartManually: false // Disable button if permission denied etc.
        });
        return null; // Indicate failure
    }
}

async function populateDevices() {
    console.log("Populating video devices...");
    try {
        await navigator.mediaDevices.getUserMedia({ video: true }); // Request permission early if needed
    } catch(e) {
        console.warn("Initial permission request failed (might be expected):", e.name);
        // Don't stop here, enumerateDevices might still work or fail later
    }

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        console.log(`Found ${videoDevices.length} video devices.`);
        updateState({ videoDevices: videoDevices });
        return videoDevices;
    } catch (error) {
        console.error("Error enumerating devices:", error);
        updateState({ statusMessage: "Error listing cameras.", error: "Could not get camera list." });
        return [];
    }
}

function startSendingFrames(video) {
    if (state.sendInterval) {
        console.log("Frame sending interval already active.");
        return; // Already sending
    }
    if (!video || video.readyState < video.HAVE_METADATA) {
        console.warn("Video element not ready for sending frames.");
        return;
    }

    const canvas = document.createElement('canvas');
    // Match canvas to video dimensions - use video's actual dimensions after loadedmetadata
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    const frameRate = 10; // Target FPS

    console.log(`Starting frame sending interval (${frameRate} FPS). Canvas: ${canvas.width}x${canvas.height}`);

    state.sendInterval = setInterval(() => {
        // Ensure WS is open and video has data
        if (state.ws && state.ws.readyState === WebSocket.OPEN && video.readyState >= video.HAVE_CURRENT_DATA) {
             try {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                canvas.toBlob(blob => {
                    if (blob && state.ws && state.ws.readyState === WebSocket.OPEN) {
                         // Check size before sending (optional, for debugging)
                         // console.log(`Sending blob size: ${blob.size}`);
                         if (blob.size > 0) {
                            state.ws.send(blob);
                         } else {
                            console.warn("Generated blob size is 0, not sending.");
                         }
                    }
                }, 'image/jpeg', 0.5); // Quality 0.5
             } catch (drawError) {
                 console.error("Error drawing video to canvas:", drawError);
                 // Consider stopping interval if error persists
             }
        } else if (state.ws && state.ws.readyState !== WebSocket.OPEN) {
             // console.warn("WebSocket not open, pausing frame send.");
             // Optional: stop interval if WS closes, handled by onclose now
        } else if (video.readyState < video.HAVE_CURRENT_DATA) {
            // console.warn("Video not ready, skipping frame send.");
        }
    }, 1000 / frameRate); // Interval based on FPS

     updateState({ isStreaming: true }); // Update state once interval is set
}

function stopStreaming() {
    console.log("Stopping streaming...");
    if (state.sendInterval) {
        clearInterval(state.sendInterval);
        state.sendInterval = null;
        console.log("Frame sending interval cleared.");
    }
    if (state.currentStream) {
        state.currentStream.getTracks().forEach(track => track.stop());
        console.log("Media stream tracks stopped.");
    }
    // Clear video source via state update
    updateState({
        currentStream: null,
        isStreaming: false,
        statusMessage: 'Streaming stopped.',
        detections: [], // Clear detections when stopping
        selectedId: null,
        selectedCardDetails: null,
        svgInstance: state.svgInstance // Preserve SVG instance if needed
    });
     // Clear SVG overlay manually if desired when stopping stream explicitly
     if(state.svgInstance) {
         state.svgInstance.clear();
     }
     console.log("Streaming stopped successfully.");
}

// ======================================================================== //
// Drawing Functions (Interacts with SVG.js instance)                      //
// ======================================================================== //

function drawDetections() {
    if (!state.svgInstance) return; // Don't draw if SVG instance isn't ready

    state.svgInstance.clear(); // Clear previous drawings using SVG.js
    const detectionsToDraw = state.detections || []; // Ensure it's an array

    detectionsToDraw.forEach(det => {
        if (!det || !det.points || !Array.isArray(det.points)) {
            console.warn("Skipping invalid detection object:", det);
            return; // Skip invalid detection data
        }

        const isSelected = det.id === state.selectedId;
        const strokeColor = isSelected ? 'yellow' : (det.color || 'lime'); // Default color if missing
        const strokeWidth = isSelected ? 4 : 2;

        // Ensure points are valid numbers
        const validPoints = det.points.filter(p => Array.isArray(p) && p.length === 2 && !isNaN(p[0]) && !isNaN(p[1]));
        if (validPoints.length !== det.points.length) {
            console.warn(`Invalid points found in detection ID ${det.id}, drawing with valid points only.`);
        }
        if(validPoints.length < 3) { // Need at least 3 points for a polygon
             console.warn(`Skipping detection ID ${det.id} due to insufficient valid points (<3).`);
             return;
        }
        const pointsString = validPoints.map(p => p.join(',')).join(' ');

        try {
            const polygon = state.svgInstance.polygon(pointsString)
                .fill('none')
                .stroke({ color: strokeColor, width: strokeWidth })
                .attr('pointer-events', 'auto') // Enable clicking on the polygon
                .on('click', (event) => {
                    event.stopPropagation(); // Prevent potential parent clicks
                    console.log(`Polygon clicked, ID: ${det.id}`);
                    handleDetectionClick(det);
                });

            const bestMatch = det.matches && det.matches.length > 0 ? det.matches[0] : null;
            if (bestMatch && bestMatch.name) {
                // Find the topmost point for label placement
                const topPoint = validPoints.reduce((a, b) => a[1] < b[1] ? a : b);
                 if (topPoint) {
                    state.svgInstance.text(bestMatch.name)
                        .move(topPoint[0], topPoint[1] - 15) // Position above the top point
                        .font({ fill: 'white', size: 12, family: "'Courier New', Courier, monospace", anchor: 'start' }) // Style text
                        .attr('pointer-events', 'none'); // Prevent text from interfering
                 }
            }
        } catch (svgError) {
            console.error(`Error drawing SVG for detection ID ${det.id}:`, svgError);
            console.error("Detection data:", det);
            console.error("Points string:", pointsString);
        }
    });
}

// Update overlay size based on video element dimensions
function updateOverlaySize() {
    const video = document.getElementById('video');
    const svgElement = document.getElementById('overlay'); // Get the actual SVG element
    if (video && svgElement && state.svgInstance) {
        const { videoWidth, videoHeight } = video; // Use intrinsic video size
        const { width, height } = video.getBoundingClientRect(); // Use display size for viewbox scaling
         // console.log(`Updating overlay: Video intrinsic ${videoWidth}x${videoHeight}, Display ${width}x${height}`);

        if (videoWidth > 0 && videoHeight > 0) {
             // Set the viewbox to match the video's intrinsic resolution
             state.svgInstance.viewbox(0, 0, videoWidth, videoHeight);
             // Ensure SVG element scales correctly within its container
             svgElement.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        } else {
            // Fallback if video dimensions aren't available yet
            state.svgInstance.viewbox(0, 0, 640, 480); // Default or last known good
        }
         // No need to set width/height attributes on SVG element itself if using viewbox properly
    }
}

// ======================================================================== //
// Event Handlers (Update state and trigger re-render)                   //
// ======================================================================== //

function handleDetectionClick(detection) {
    console.log("Handling click on detection ID:", detection.id);
    const bestMatch = detection.matches && detection.matches.length > 0 ? detection.matches[0] : null;
    updateState({
        selectedId: detection.id,
        selectedCardDetails: bestMatch // Store the details directly
    });
    // Re-rendering will update the sidebar and redraw detections with new selection highlight
}

function handleSidebarItemClick(detection) {
     console.log("Handling click on sidebar item ID:", detection.id);
     const bestMatch = detection.matches && detection.matches.length > 0 ? detection.matches[0] : null;
     updateState({
         selectedId: detection.id,
         selectedCardDetails: bestMatch // Store the details directly
     });
     // Re-rendering will update the selection highlight in sidebar and SVG
}

async function handleDeviceChange(event) {
    const newDeviceId = event.target.value;
    console.log(`Device selection changed to: ${newDeviceId}`);
    updateState({ statusMessage: `Switching camera to ${event.target.options[event.target.selectedIndex].text}...`});
    // Stop existing frame sending before changing stream
    if (state.sendInterval) {
        clearInterval(state.sendInterval);
        state.sendInterval = null;
    }
    const newStream = await startStream(newDeviceId);
    if (newStream) {
        updateState({ statusMessage: 'Camera changed. Streaming...' });
        // Restart frame sending *after* the new stream is ready and video element is updated
        const videoElement = document.getElementById('video');
        if(videoElement) {
             videoElement.addEventListener('playing', () => {
                  console.log("New stream playing, starting frame sending.");
                  startSendingFrames(videoElement);
             }, { once: true });
        } else {
             console.error("Cannot restart frame sending, video element not found after device change.");
             updateState({statusMessage: "Error restarting stream.", error: "Video element not found."});
        }

    } else {
        updateState({ statusMessage: 'Failed to switch camera.', isStreaming: false });
    }
}

async function handleStartStopClick() {
    const videoElement = document.getElementById('video');
    if (!videoElement) {
         console.error("Start/Stop click failed: Video element not found.");
         updateState({statusMessage: "Error: Video element missing.", error: "Video element not found."});
         return;
    }

    if (state.isStreaming) {
        // Stop Streaming
        stopStreaming();
        // Button text/state updated via re-render
    } else {
        // Start Streaming
        updateState({ statusMessage: 'Requesting camera access...', isAutoStarting: false }); // Clear auto-start flag

        const selectedDeviceId = state.currentDeviceId || (state.videoDevices.length > 0 ? state.videoDevices[0].deviceId : null);
        const stream = await startStream(selectedDeviceId); // Use current/first device

        if (stream) {
             // Wait for video to start playing before sending frames
             videoElement.addEventListener('playing', () => {
                 console.log("Manual start: Video playing, starting frame sending.");
                 updateState({ statusMessage: 'Streaming to server...' });
                 startSendingFrames(videoElement);
             }, { once: true });
        } else {
            // Error handled within startStream, state already updated
            console.log("Manual start failed.");
        }
    }
}

// ======================================================================== //
// lit-html Template                                                        //
// ======================================================================== //

const appTemplate = (state) => html`
    <div id="video-container">
        <video id="video" autoplay muted playsinline .srcObject=${state.currentStream}></video>
        <svg id="overlay"></svg>
    </div>

    <div id="sidebar">
        <div id="controls">
            <select id="select" @change=${handleDeviceChange} ?disabled=${state.isAutoStarting || state.videoDevices.length === 0}>
                ${state.videoDevices.length === 0 ? html`<option>No cameras found</option>` : ''}
                ${state.videoDevices.map(device => html`
                    <option value=${device.deviceId} ?selected=${device.deviceId === state.currentDeviceId}>
                        ${device.label || `Camera ${state.videoDevices.indexOf(device) + 1}`}
                    </option>
                `)}
            </select>
            <button
                id="startCamera"
                @click=${handleStartStopClick}
                ?disabled=${state.isAutoStarting || !state.canStartManually}>
                ${state.isStreaming ? 'Stop Streaming' : 'Start Streaming'}
            </button>
            <div id="status">${state.statusMessage}</div>
            ${state.error ? html`<div style="color: red; margin-top: 5px;">${state.error}</div>` : ''}
        </div>

        <div id="card-list">
            ${state.detections.length === 0 && !state.isStreaming && !state.isAutoStarting
                ? html`<p>Streaming stopped or no cards detected.</p>`
                : state.detections.length === 0 && (state.isStreaming || state.isAutoStarting)
                ? html`<p>Scanning for cards...</p>`
                : html`
                    ${[...state.detections] // Create a copy before sorting
                        .sort((a, b) => (a.id || 0) - (b.id || 0)) // Sort by ID
                        .map(det => {
                            const bestMatch = det.matches && det.matches.length > 0 ? det.matches[0] : null;
                            const isSelected = det.id === state.selectedId;
                            return html`
                                <div
                                    class="card-list-item ${isSelected ? 'selected' : ''}"
                                    @click=${() => handleSidebarItemClick(det)}>
                                    <div class="img-container">
                                        <img src=${'data:image/jpeg;base64,' + det.img} alt="Detected Card Crop">
                                        ${bestMatch && bestMatch.img_uri
                                            ? html`<img class="uri-img" src=${bestMatch.img_uri} alt="Best Match Image">`
                                            : ''}
                                    </div>
                                    <div class="info">
                                        <strong>ID: ${det.id}</strong><br>
                                        ${bestMatch ? bestMatch.name : 'Unknown'}
                                    </div>
                                </div>`;
                         })}
                `}
        </div>

        <div id="card-info">
            ${state.selectedCardDetails ? html`
                <h3>${state.selectedCardDetails.name || 'Unknown Card'}</h3>
                <p>
                    Set: ${state.selectedCardDetails.set_name || 'Unknown'}
                    (${state.selectedCardDetails.set_code || 'N/A'})
                </p>
                <img
                    src=${state.selectedCardDetails.img_uri /* || 'data:image/jpeg;base64,' + findDetectionImage(state.selectedId, state.detections) */}
                    alt=${state.selectedCardDetails.name || 'Selected Card'}
                    @error=${(e) => e.target.style.display='none'} /* Hide if image fails to load */
                >
                ` : state.selectedId !== null ? html`
                 <p>Loading details for ID ${state.selectedId}...</p>
                 ` : html`
                 <p>Click a detected card (in video or list) to see details.</p>
            `}
        </div>
    </div>
`;


// ======================================================================== //
// State Update & Rendering Function                                        //
// ======================================================================== //

// Central function to update state and trigger re-render
function updateState(newState) {
    state = { ...state, ...newState };
    // Schedule microtask for rendering to batch updates
    Promise.resolve().then(() => {
         const container = document.getElementById('app-container');
         if(container) {
            render(appTemplate(state), container);
            // Drawing needs to happen *after* render ensures SVG element exists
            // We might need to defer drawing slightly or ensure SVG instance is valid
            if (state.svgInstance) {
                 drawDetections(); // Redraw SVG overlay based on new state
            } else if(document.getElementById('overlay')) {
                 // Initialize SVG.js if overlay exists but instance is not set yet
                 initializeSvg();
                 if(state.svgInstance) drawDetections();
            }
         } else {
             console.error("App container not found for rendering!");
         }

    });
}

// ======================================================================== //
// Initialization Function                                                  //
// ======================================================================== //
function initializeSvg() {
     const overlayElement = document.getElementById('overlay');
     if (overlayElement && !state.svgInstance) {
         console.log("Initializing SVG.js instance.");
         const instance = SVG(overlayElement); // Use the ID of the SVG element
         if(instance) {
            updateState({ svgInstance: instance });
            updateOverlaySize(); // Set initial size
         } else {
            console.error("SVG.js failed to initialize on #overlay");
         }
     } else if(state.svgInstance) {
         // console.log("SVG.js already initialized.");
     } else {
         console.warn("SVG overlay element not found yet for initialization.");
     }
}


async function main() {
    console.log("Main function started.");
    // Initial Render
    updateState({ statusMessage: 'Initializing...' }); // Render initial UI

    // Need to wait for first render to potentially access DOM elements like video/overlay
    await Promise.resolve(); // Wait for microtask queue (where render happens)

     initializeSvg(); // Attempt to init SVG.js after first render

    // Setup WebSocket
    connectWebSocket();

    // Populate devices early
    await populateDevices();

    // Add listener after devices are populated
     window.addEventListener('resize', updateOverlaySize);

    // Attempt to auto-start the stream
    updateState({ isAutoStarting: true, statusMessage: 'Attempting auto-start...', canStartManually: false });
    const stream = await startStream(null); // Try default camera first

    if (stream) {
        console.log("Auto-start successful.");
         const videoElement = document.getElementById('video');
         if (videoElement) {
            // Wait for video to be ready before starting frame sending
            videoElement.addEventListener('playing', () => {
                 console.log("Auto-start: Video playing, starting frame sending.");
                 updateOverlaySize(); // Ensure overlay is correct size now
                 startSendingFrames(videoElement);
                 updateState({ isAutoStarting: false, statusMessage: 'Streaming to server...', canStartManually: true });
            }, { once: true });
             videoElement.addEventListener('loadedmetadata', () => {
                updateOverlaySize(); // Update size when metadata loads too
                // Re-populate devices using the actual device ID from the stream
                const actualDeviceId = stream.getVideoTracks()[0]?.getSettings()?.deviceId;
                if (actualDeviceId) {
                    updateState({ currentDeviceId: actualDeviceId });
                    // No need to call populateDevices again unless list might change
                }
            }, { once: true });
         } else {
             console.error("Auto-start: Video element not found after stream success.");
             stopStreaming(); // Clean up stream
             updateState({ isAutoStarting: false, statusMessage: 'Auto-start failed (internal error).', error: "Video element missing.", canStartManually: true });
         }
    } else {
        console.warn('Auto-start failed. User interaction likely required.');
        // Error messages are set within startStream
        updateState({ isAutoStarting: false, canStartManually: !state.error }); // Enable button if no blocking error
    }
}

// Start the application
main();
