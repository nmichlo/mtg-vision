// ========== Global Variables ==========
let currentStream;
let ws;
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
        console.log('Detections received:', data.detections); // Debug log
        drawDetections(data.detections); // Update overlay with D3
        updateSidebar(data.detections);  // Update sidebar with Alpine
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
    try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        document.getElementById('video').srcObject = currentStream;
    } catch (error) {
        console.error('Error starting stream:', error);
        document.getElementById('status').textContent = 'Error: Could not access camera.';
        throw error;
    }
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
    const svg = d3.select('#overlay');
    const cardList = document.getElementById('card-list');
    const selectedId = cardList.__x?.$data?.selectedId ?? null;

    // Bind data to polygons
    const polygons = svg.selectAll('polygon')
        .data(detections, d => d.id);

    polygons.exit().remove();

    const polygonsEnter = polygons.enter()
        .append('polygon')
        .attr('fill', 'none')
        .style('pointer-events', 'auto')
        .on('click', (event, d) => {
            if (cardList.__x?.$data) {
                cardList.__x.$data.selectedId = d.id;
                cardList.__x.$data.$dispatch('card-selected', d);
            }
        });

    polygonsEnter.merge(polygons)
        .attr('points', d => d.points.map(p => p.join(',')).join(' '))
        .attr('stroke', d => d.id === selectedId ? 'yellow' : d.color)
        .attr('stroke-width', d => d.id === selectedId ? '4' : '2');

    // Bind data to text labels
    const texts = svg.selectAll('text')
        .data(detections.filter(d => d.matches[0]), d => d.id);

    texts.exit().remove();

    texts.enter()
        .append('text')
        .attr('fill', 'white')
        .attr('font-size', '12')
        .merge(texts)
        .attr('x', d => d.points.reduce((a, b) => a[1] < b[1] ? a : b)[0])
        .attr('y', d => d.points.reduce((a, b) => a[1] < b[1] ? a : b)[1] - 5)
        .text(d => d.matches[0].name);
}

function updateSidebar(detections) {
    const cardList = document.getElementById('card-list');
    const sortedDets = [...detections].sort((a, b) => a.id - b.id);
    if (cardList.__x?.$data) {
        // Clear existing detections and push new ones to trigger reactivity
        cardList.__x.$data.detections.splice(0, cardList.__x.$data.detections.length);
        sortedDets.forEach(det => cardList.__x.$data.detections.push(det));
    } else {
        console.warn('Alpine.js not initialized yet; retrying in 100ms');
        setTimeout(() => updateSidebar(detections), 100);
    }
}

function updateOverlaySize() {
    const video = document.getElementById('video');
    const svg = document.getElementById('overlay');
    svg.setAttribute('width', video.clientWidth);
    svg.setAttribute('height', video.clientHeight);
    svg.setAttribute('viewBox', '0 0 640 480');
}

// ========== Main Function ==========
async function main() {
    const video = document.getElementById('video');
    const select = document.getElementById('select');
    const status = document.getElementById('status');
    const startCameraButton = document.getElementById('startCamera');

    // Wait for Alpine.js to initialize
    await new Promise(resolve => {
        document.addEventListener('alpine:init', () => {
            console.log('Alpine.js initialized');
            resolve();
        });
    });

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
        startCameraButton.disabled = true;
    } catch (error) {
        console.warn('Auto-start failed:', error);
        status.textContent = 'Auto-start failed. Click "Start Streaming" to begin.';
    }

    // Manual start on button click
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

    // Populate device list on change
    select.addEventListener('change', async () => {
        try {
            await startStream(select.value);
        } catch (error) {
            status.textContent = 'Error: Could not switch camera.';
            startCameraButton.disabled = false;
        }
    });

    // Update overlay size on window resize
    window.addEventListener('resize', updateOverlaySize);
    updateOverlaySize(); // Initial call
}
