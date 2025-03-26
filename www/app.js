// Global variables
let currentStream;
const inputCanvas = document.createElement('canvas');
inputCanvas.width = 640;
inputCanvas.height = 640;
const inputCtx = inputCanvas.getContext('2d');

// Start video stream with the specified device ID
async function startStream(deviceId) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    const constraints = {
        video: { deviceId: { exact: deviceId } }
    };
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    document.getElementById('video').srcObject = currentStream;
}

// Populate the device selector dropdown
async function populateDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    const select = document.getElementById('select');
    select.innerHTML = '';
    videoDevices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index + 1}`;
        select.appendChild(option);
    });
    if (videoDevices.length > 0) {
        await startStream(videoDevices[0].deviceId);
    }
}

// Process the ONNX model output (assuming [1, num_boxes, 6] format: [x1, y1, x2, y2, confidence, class])
function processOutput(outputTensor) {
    const data = outputTensor.data;
    const detections = [];
    const numBoxes = outputTensor.dims[1];
    for (let i = 0; i < numBoxes; i++) {
        const offset = i * 6;
        const x1 = data[offset];
        const y1 = data[offset + 1];
        const x2 = data[offset + 2];
        const y2 = data[offset + 3];
        const confidence = data[offset + 4];
        const classId = data[offset + 5];
        if (confidence > 0.5) { // Confidence threshold
            detections.push({ x1, y1, x2, y2, confidence, classId });
        }
    }
    return detections;
}

// Draw bounding boxes on the SVG overlay
function drawBoundingBoxes(detections) {
    const svg = document.getElementById('overlay');
    svg.innerHTML = ''; // Clear previous boxes
    const padding = (640 - 480) / 2; // Padding added to make 640x480 into 640x640
    detections.forEach(det => {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        const x = det.x1;
        const y = det.y1 - padding; // Adjust for padding
        const width = det.x2 - det.x1;
        const height = (det.y2 - padding) - y;
        // Skip if box is outside video area
        if (y < 0 || y + height > 480) return;
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', width);
        rect.setAttribute('height', height);
        rect.setAttribute('stroke', 'red');
        rect.setAttribute('stroke-width', '2');
        rect.setAttribute('fill', 'none');
        svg.appendChild(rect);
    });
}

function fillArray(data, imageData) {
    const w = imageData.width
    const h = imageData.height
    const c = imageData.data.length / w / h;
    const mx = 1
    const my = mx * w;
    const mc = my * h;
    console.log(w, h, c, mx, my, mc)
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const idx = (y * w + x) * c;
            data[0 * mc + y * my + x] = imageData.data[idx + 0] / 255; // R
            data[1 * mc + y * my + x] = imageData.data[idx + 1] / 255; // G
            data[2 * mc + y * my + x] = imageData.data[idx + 2] / 255; // B
        }
    }
}

async function detect(session, video) {
    const VIDEO_H = 480;
    const VIDEO_W = 640;

    const MODEL_H = 640;
    const MODEL_W = 640;
    const MODEL_C = 3;

    // TODO get video size instead of using magix vars
    // Prepare 640x640 input with padding
    inputCtx.fillStyle = 'black';
    inputCtx.fillRect(0, 0, MODEL_W, MODEL_H);
    inputCtx.drawImage(video, 0, (640 - 480) / 2, 640, 480);

    // Convert to tensor
    const imageData = inputCtx.getImageData(0, 0, MODEL_W, MODEL_H);
    const data = new Float32Array(MODEL_C * MODEL_W * MODEL_H);
    fillArray(data, imageData)
    const inputTensor = new ort.Tensor('float32', data, [1, MODEL_C, MODEL_H, MODEL_W]);

    // Run inference
    console.log(inputTensor)
    console.log(session)
    const outputMap = await session.run({ images: inputTensor });
    console.log(outputMap)
    const outputTensor = outputMap.output0.cpuData;  // outputMap.output0.dims == [1, 300, 7]

    // TODO: nms doesn't seem to be supported....


    // Process and display results
    const detections = processOutput(outputTensor);
    return detections;
}

// Inference loop to process video frames
async function inferenceLoop(session) {
    const video = document.getElementById('video');
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        const detections = await detect(session, video);
        drawBoundingBoxes(detections);
    }
    requestAnimationFrame(() => inferenceLoop(session));
}

// Main function
async function main() {
    const video = document.getElementById('video');
    const select = document.getElementById('select');
    const status = document.getElementById('status');

    // Load the ONNX model
    status.textContent = 'Loading model...';
    const session = await ort.InferenceSession.create('1ipho2mn_best.onnx');
    status.textContent = 'Model loaded.';

    // Populate devices and start the stream
    await populateDevices();

    // Handle device selection changes
    select.addEventListener('change', async () => {
        const deviceId = select.value;
        await startStream(deviceId);
    });

    // Update devices when new ones are plugged in
    navigator.mediaDevices.addEventListener('devicechange', populateDevices);

    // Start the inference loop
    inferenceLoop(session);
}
