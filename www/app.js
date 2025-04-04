// ======================================================================== //
// Global Vars                                                              //
// ======================================================================== //

let currentStream;
let ws;
let selectedId = null;
let reconnectTimeout;
let svg; // SVG.js instance
let isStreaming = false; // Track streaming state
let sendInterval; // Store interval for sending frames

// ======================================================================== //
// WebSocket Handling                                                       //
// ======================================================================== //

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/detect`;

    ws = new WebSocket(wsUrl);

    // setup event handlers
    ws.onopen = () => {
        console.log('WebSocket connection established');
        document.getElementById('status').textContent = 'Connected to server.';
    };
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        drawDetections(data.detections);
        updateSidebar(data.detections);
    };
    ws.onerror = () => {
        console.error('WebSocket error');
        document.getElementById('status').textContent = 'WebSocket error occurred.';
    };
    ws.onclose = () => {
        console.log('WebSocket connection closed');
        document.getElementById('status').textContent = 'Disconnected from server. Reconnecting...';
        reconnectTimeout = setTimeout(connectWebSocket, 5000);
    };
}


// ======================================================================== //
// Video Stream Handling                                                    //
// ======================================================================== //


async function startStream(deviceId) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    const constraints = { video: { deviceId: { exact: deviceId }, width: 640, height: 480 } };
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    document.getElementById('video').srcObject = currentStream;
}

async function populateDevices(currentDeviceId) {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    const select = document.getElementById('select');
    select.innerHTML = '';
    videoDevices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index + 1}`;
        if (device.deviceId === currentDeviceId) {
            option.selected = true;
        }
        select.appendChild(option);
    });
}

function startSendingFrames(video) {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');

    sendInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ctx.drawImage(video, 0, 0, 640, 480);
            canvas.toBlob(blob => {
                ws.send(blob);
            }, 'image/jpeg', 0.5);
        }
    }, 100); // 10 FPS
}

function stopStreaming() {
    if (sendInterval) {
        clearInterval(sendInterval);
        sendInterval = null;
    }
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    document.getElementById('video').srcObject = null;
    isStreaming = false;
    document.getElementById('startCamera').textContent = 'Start Streaming';
    document.getElementById('status').textContent = 'Streaming stopped.';
}

// ======================================================================== //
// Drawing Functions                                                        //
// ======================================================================== //


function drawDetections(detections) {
    svg.clear(); // Clear previous drawings using SVG.js
    detections.forEach(det => {
        const isSelected = det.id === selectedId;
        const strokeColor = isSelected ? 'yellow' : det.color;
        const strokeWidth = isSelected ? 4 : 2;
        const points = det.points.map(p => p.join(',')).join(' ');

        const polygon = svg.polygon(points)
            .fill('none')
            .stroke({ color: strokeColor, width: strokeWidth })
            .attr('pointer-events', 'auto') // Enable clicking on the polygon
            .on('click', () => {
                selectedId = det.id;
                updateCardInfo(det);
                updateSidebar(detections); // Refresh sidebar
            });

        const bestMatch = det.matches[0];
        if (bestMatch) {
            const topPoint = det.points.reduce((a, b) => a[1] < b[1] ? a : b);
            svg.text(bestMatch.name)
                .move(topPoint[0], topPoint[1] - 15)
                .font({ fill: 'white', size: 12 })
                .attr('pointer-events', 'none'); // Prevent text from interfering with polygon clicks
        }
    });
}

function updateSidebar(detections) {
    console.log('Detections received for sidebar:', detections); // Debugging
    const cardList = document.getElementById('card-list');
    cardList.innerHTML = '';
    if (detections.length === 0) {
        cardList.innerHTML = '<p>No cards detected</p>';
        return;
    }
    const sortedDets = [...detections].sort((a, b) => a.id - b.id);

    sortedDets.forEach(det => {
        const div = document.createElement('div');
        div.style.display = 'flex';
        div.style.alignItems = 'center';
        div.style.marginBottom = '10px';
        div.style.cursor = 'pointer';
        div.classList.add('card-list-item'); // Add class for hover effect
        if (det.id === selectedId) div.style.border = '2px solid yellow';

        const imgContainer = document.createElement('div');
        imgContainer.style.position = 'relative';
        imgContainer.style.width = '50px';
        imgContainer.style.height = '70px'; // Adjust height as needed
        imgContainer.style.overflow = 'hidden';
        imgContainer.style.borderRadius = '3px';

        const img = document.createElement('img');
        img.src = 'data:image/jpeg;base64,' + det.img;
        img.style.width = '100%';
        img.style.height = 'auto';
        imgContainer.appendChild(img);

        const uriImg = document.createElement('img');
        const bestMatch = det.matches[0];
        uriImg.src = bestMatch ? bestMatch.img_uri : '';
        uriImg.style.position = 'absolute';
        uriImg.style.top = '0';
        uriImg.style.left = '0';
        uriImg.style.width = '100%';
        uriImg.style.height = '100%';
        uriImg.style.objectFit = 'cover';
        if (bestMatch && bestMatch.img_uri) {
            imgContainer.appendChild(uriImg);
        }

        div.appendChild(imgContainer);

        const info = document.createElement('div');
        info.style.marginLeft = '10px';
        const bestMatchName = bestMatch ? bestMatch.name : 'Unknown';
        info.innerHTML = `<strong>ID: ${det.id}</strong><br>${bestMatchName}`;
        div.appendChild(info);

        div.addEventListener('click', () => {
            selectedId = det.id;
            updateCardInfo(det);
            updateSidebar(detections);
        });
        cardList.appendChild(div);
    });
}

function updateCardInfo(det) {
    const cardInfo = document.getElementById('card-info');
    if (!det) {
        cardInfo.innerHTML = '';
        return;
    }
    const bestMatch = det.matches[0];
    cardInfo.innerHTML = `
        <h3>${bestMatch.name}</h3>
        <p>Set: ${bestMatch.set_name || 'Unknown'} (${bestMatch.set_code || ''})</p>
        <img src="${bestMatch.img_uri || 'data:image/jpeg;base64,' + det.img}" alt="${bestMatch.name}" style="width: 100%; height: auto; border-radius: 3px;">
    `;
    console.log(bestMatch)
    console.log(bestMatch)
    console.log(bestMatch)
}

function updateOverlaySize() {
    const video = document.getElementById('video');
    const { width, height } = video.getBoundingClientRect();
    svg.viewbox(0, 0, width, height); // Update viewBox with SVG.js
}


// ======================================================================== //
// Main Function                                                            //
// ======================================================================== //

async function main() {
    const video = document.getElementById('video');
    const select = document.getElementById('select');
    const status = document.getElementById('status');
    const startCameraButton = document.getElementById('startCamera');
    const videoContainer = document.getElementById('video-container');
    const overlay = document.getElementById('overlay');

    // Initialize WebSocket and SVG overlay
    connectWebSocket();
    svg = SVG(overlay);

    // helper function to initialize to start streaming
    async function start() {
        const initialStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = initialStream;
        currentStream = initialStream;
        const deviceId = initialStream.getVideoTracks()[0].getSettings().deviceId;
        status.textContent = 'Camera access granted. Populating devices...';
        await populateDevices(deviceId);
        status.textContent = 'Streaming to server...';
        startSendingFrames(video);
    }

    // Attempt to auto-start the stream
    try {
        await start();
        startCameraButton.disabled = true; // Disable button since stream is active
    } catch (error) {
        console.warn('Auto-start failed:', error);
        status.textContent = 'Auto-start failed. Click "Start Streaming" to begin.';
    }

    // Manual start on button click (for browsers requiring interaction)
    startCameraButton.addEventListener('click', async () => {
        const isStarted = startCameraButton.textContent === 'Stop Streaming';

        if (!isStarted) {
            startCameraButton.disabled = true;
            status.textContent = 'Requesting camera access...';
            try {
                startCameraButton.textContent = 'Stop Streaming';
                await start();
            } catch (error) {
                console.error('Error accessing camera:', error);
                startCameraButton.textContent = 'Start Streaming';
                status.textContent = 'Error: Could not access camera.';
            }
            startCameraButton.disabled = false;
        } else {
            stopStreaming();
            startCameraButton.textContent = 'Start Streaming';
            startCameraButton.disabled = false;
        }
    });

    // Populate device list
    select.addEventListener('change', async () => {
        await startStream(select.value);
    });

    // Update overlay size on window resize
    window.addEventListener('resize', updateOverlaySize);
    updateOverlaySize(); // Initial call

    // Ensure overlay size is updated when video loads (in case of auto-start)
    video.addEventListener('loadedmetadata', updateOverlaySize);
}
