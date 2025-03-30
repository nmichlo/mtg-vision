// ========== Global Variables ==========
let currentStream;
let ws;
let selectedId = null;
let reconnectTimeout;

// ========== WebSocket Handling ==========
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `ws://${window.location.host}/detect`;

    ws = new WebSocket(wsUrl);
    const loading = document.getElementById('loading');
    loading.style.display = 'block';
    ws.onopen = () => {
        console.log('WebSocket connection established');
        document.getElementById('status').textContent = 'Connected to server.';
        loading.style.display = 'none';
    };
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        drawDetections(data.detections);
        updateSidebar(data.detections);
    };
    ws.onerror = () => {
        console.error('WebSocket error');
        document.getElementById('status').textContent = 'WebSocket error occurred.';
        loading.style.display = 'block';
    };
    ws.onclose = () => {
        console.log('WebSocket connection closed');
        document.getElementById('status').textContent = 'Disconnected from server. Reconnecting...';
        loading.style.display = 'block';
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

// ========== Drawing Functions ==========
function drawDetections(detections) {
    const svg = document.getElementById('overlay');
    svg.innerHTML = '';
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
        polygon.addEventListener('click', () => { selectedId = det.id; });
        svg.appendChild(polygon);

        const topPoint = det.points.reduce((a, b) => a[1] < b[1] ? a : b);
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', topPoint[0]);
        text.setAttribute('y', topPoint[1] - 5);
        text.setAttribute('fill', 'white');
        text.setAttribute('font-size', '12');
        text.textContent = det.match.name;
        svg.appendChild(text);
    });
}

function updateSidebar(detections) {
    const sidebar = document.getElementById('sidebar');
    sidebar.innerHTML = '';
    const sortedDets = [...detections].sort((a, b) => a.id - b.id);
    sortedDets.forEach(det => {
        const div = document.createElement('div');
        div.style.marginBottom = '10px';
        if (det.id === selectedId) div.style.border = '2px solid yellow';
        const colorDiv = document.createElement('div');
        colorDiv.style.width = '20px';
        colorDiv.style.height = '20px';
        colorDiv.style.backgroundColor = det.color;
        colorDiv.style.display = 'inline-block';
        div.appendChild(colorDiv);
        const img = document.createElement('img');
        img.src = 'data:image/jpeg;base64,' + det.img;
        img.style.width = '100px';
        img.style.height = 'auto';
        div.appendChild(img);
        const name = document.createElement('p');
        name.textContent = det.match.name;
        div.appendChild(name);
        div.addEventListener('click', () => { selectedId = det.id; });
        sidebar.appendChild(div);
    });
}

// ========== Main Function ==========
async function main() {
    const video = document.getElementById('video');
    const select = document.getElementById('select');
    const status = document.getElementById('status');
    const startCameraButton = document.getElementById('startCamera');

    connectWebSocket();

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

    select.addEventListener('change', async () => {
        await startStream(select.value);
    });
}
