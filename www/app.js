// Global variables
let currentStream;
let ws;

// Start video stream with the specified device ID
async function startStream(deviceId) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    const constraints = { video: { deviceId: { exact: deviceId }, width: 640, height: 480 } };
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    document.getElementById('video').srcObject = currentStream;
}

// Populate the device selector dropdown
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

// Draw bounding boxes on the SVG overlay
function drawBoundingBoxes(detections) {
    const svg = document.getElementById('overlay');
    svg.innerHTML = ''; // Clear previous boxes
    detections.forEach(det => {
        // Convert points array to SVG points string (e.g., "x1,y1 x2,y2 x3,y3 x4,y4")
        const pointsStr = det.points.map(p => p.join(',')).join(' ');
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', pointsStr);
        polygon.setAttribute('stroke', 'red');
        polygon.setAttribute('stroke-width', '2');
        polygon.setAttribute('fill', 'none');
        svg.appendChild(polygon);
    });
}

// Send video frames to the backend over WebSocket
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
            }, 'image/jpeg', 0.5); // 50% quality to reduce bandwidth
        }
    }, 100); // Send at 10 FPS (every 100ms)
}

// Main function
async function main() {
    const video = document.getElementById('video');
    const select = document.getElementById('select');
    const status = document.getElementById('status');
    const startCameraButton = document.getElementById('startCamera');

    // Initialize WebSocket connection
    ws = new WebSocket('ws://localhost:8000/detect');
    ws.onopen = () => {
        console.log('WebSocket connection established');
        status.textContent = 'Connected to server. Select a camera to start.';
    };
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        drawBoundingBoxes(data.detections);
    };
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        status.textContent = 'WebSocket error occurred.';
    };
    ws.onclose = () => {
        console.log('WebSocket connection closed');
        status.textContent = 'Disconnected from server.';
    };

    // Start camera on button click
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
            startSendingFrames(video); // Start sending frames to the backend
        } catch (error) {
            console.error('Error accessing camera:', error);
            status.textContent = 'Error: Could not access camera.';
            startCameraButton.disabled = false;
        }
    });

    // Handle device selection changes
    select.addEventListener('change', async () => {
        const deviceId = select.value;
        await startStream(deviceId);
    });
}
