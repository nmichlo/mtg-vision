// ========== Global Variables ==========
let currentStream;
let ws;
let selectedId = null;
let reconnectTimeout;

// ========== WebSocket Handling ==========
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/detect`;

    ws = new WebSocket(wsUrl);
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

// ========== Video Stream Handling ==========
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

    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ctx.drawImage(video, 0, 0, 640, 480);
            canvas.toBlob(blob => {
                ws.send(blob);
            }, 'image/jpeg', 0.5);
        }
    }, 100); // 10 FPS
}

// ========== Drawing Functions ==========
function drawDetections(detections) {
    const svg = document.getElementById('overlay');
    svg.innerHTML = ''; // Clear previous drawings
    detections.forEach(det => {
        const isSelected = det.id === selectedId;
        const strokeColor = isSelected ? 'yellow' : det.color;
        const strokeWidth = isSelected ? '4' : '2';
        const pointsStr = det.points.map(p => p.join(',')).join(' ');
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', pointsStr);
        polygon.setAttribute('stroke', strokeColor);
        polygon.setAttribute('stroke-width', strokeWidth);
        polygon.setAttribute('fill', 'none');
        polygon.style.pointerEvents = 'auto'; // Enable clicking
        polygon.addEventListener('click', () => {
            selectedId = det.id;
            updateCardInfo(det);
            updateSidebar(detections); // Refresh sidebar to highlight selected card
        });
        svg.appendChild(polygon);

        // Display the best match name above the polygon
        const bestMatch = det.matches[0];
        if (bestMatch) {
            const topPoint = det.points.reduce((a, b) => a[1] < b[1] ? a : b); // Find top-most point
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', topPoint[0]);
            text.setAttribute('y', topPoint[1] - 5);
            text.setAttribute('fill', 'white');
            text.setAttribute('font-size', '12');
            text.textContent = bestMatch.name;
            svg.appendChild(text);
        }
    });
}

function updateSidebar(detections) {
    const cardList = document.getElementById('card-list');
    cardList.innerHTML = ''; // Clear previous content
    if (detections.length === 0) {
        cardList.innerHTML = '<p>No cards detected</p>';
        return;
    }
    const sortedDets = [...detections].sort((a, b) => a.id - b.id); // Sort by ID

    sortedDets.forEach(det => {
        const div = document.createElement('div');
        div.style.display = 'flex';
        div.style.alignItems = 'center';
        div.style.marginBottom = '10px';
        div.style.cursor = 'pointer';
        if (det.id === selectedId) div.style.border = '2px solid yellow'; // Highlight selected card
        const img = document.createElement('img');
        img.src = 'data:image/jpeg;base64,' + det.img;
        img.style.width = '50px';
        img.style.height = 'auto';
        div.appendChild(img);
        const info = document.createElement('div');
        info.style.marginLeft = '10px';
        const bestMatch = det.matches[0];
        info.innerHTML = `<strong>ID: ${det.id}</strong><br>${bestMatch ? bestMatch.name : 'Unknown'}`;
        div.appendChild(info);
        div.addEventListener('click', () => {
            selectedId = det.id;
            updateCardInfo(det);
            updateSidebar(detections); // Refresh to highlight selected card
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
        <img src="${bestMatch.img_uri || 'data:image/jpeg;base64,' + det.img}" alt="${bestMatch.name}" style="width: 100%; height: auto;">
    `;
}

function updateOverlaySize() {
    const video = document.getElementById('video');
    const svg = document.getElementById('overlay');
    const { width, height } = video.getBoundingClientRect();
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
}

// ========== Main Function ==========
async function main() {
    const video = document.getElementById('video');
    const select = document.getElementById('select');
    const status = document.getElementById('status');
    const startCameraButton = document.getElementById('startCamera');

    connectWebSocket();

    // Attempt to auto-start the stream
    try {
        const initialStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = initialStream;
        currentStream = initialStream;
        const deviceId = initialStream.getVideoTracks()[0].getSettings().deviceId;
        status.textContent = 'Camera access granted. Populating devices...';
        await populateDevices(deviceId);
        status.textContent = 'Streaming to server...';
        startSendingFrames(video);
        startCameraButton.disabled = true; // Disable button since stream is active
    } catch (error) {
        console.warn('Auto-start failed:', error);
        status.textContent = 'Auto-start failed. Click "Start Streaming" to begin.';
    }

    // Manual start on button click (for browsers requiring interaction)
    startCameraButton.addEventListener('click', async () => {
        startCameraButton.disabled = true;
        status.textContent = 'Requesting camera access...';
        try {
            const initialStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            video.srcObject = initialStream;
            currentStream = initialStream;
            const deviceId = initialStream.getVideoTracks()[0].getSettings().deviceId;
            status.textContent = 'Camera access granted. Populating devices...';
            await populateDevices(deviceId);
            status.textContent = 'Streaming to server...';
            startSendingFrames(video);
        } catch (error) {
            console.error('Error accessing camera:', error);
            status.textContent = 'Error: Could not access camera.';
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
}
