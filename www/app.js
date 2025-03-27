// Global variables
let currentStream;
const inputCanvas = document.createElement('canvas');
inputCanvas.width = 640;
inputCanvas.height = 640;
const inputCtx = inputCanvas.getContext('2d');

// ======================================================================== //
// START STREAM                                                             //
// ======================================================================== //

// Start video stream with the specified device ID
async function startStream(deviceId) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    const constraints = { video: { deviceId: { exact: deviceId } } };
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

// ======================================================================== //
// INFERENCE                                                                //
// ======================================================================== //

// Process the ONNX model output (assuming [1, num_boxes, 6] format: [x1, y1, x2, y2, confidence, class])
function processOutput(outputTensor) {
    const data = outputTensor.data;
    const detections = [];
    const numBoxes = outputTensor.dims[1];

    // cx: Center x-coordinate (normalized).
    // cy: Center y-coordinate (normalized).
    // w: Width of the bounding box (normalized).
    // h: Height of the bounding box (normalized).
    // theta: Rotation angle (in radians, representing the orientation of the box).
    // confidence: Objectness score (indicating the likelihood of an object being present). 7+. class_scores: One score per class (e.g., if there are nc classes, there will be nc additional values).
    // classId
    for (let i = 0; i < numBoxes; i++) {
        const offset = i * 7;
        const cx = data[offset];
        const cy = data[offset + 1];
        const w = data[offset + 2];
        const h = data[offset + 3];
        const theta = data[offset + 4];
        const confidence = data[offset + 5];
        const classId = data[offset + 6];
        if (confidence > 0.5) { // Confidence threshold
            detections.push({ cx, cy, w, h, theta, confidence, classId });
        }
    }
    return detections;
}

function fillArray(data, imageData) {
    const w = imageData.width;
    const h = imageData.height;
    const c = imageData.data.length / w / h;
    const mx = 1;
    const my = mx * w;
    const mc = my * h;
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

    // Prepare 640x640 input with padding
    inputCtx.fillStyle = 'black';
    inputCtx.fillRect(0, 0, MODEL_W, MODEL_H);
    inputCtx.drawImage(video, 0, (640 - 480) / 2, 640, 480);

    // Convert to tensor
    const imageData = inputCtx.getImageData(0, 0, MODEL_W, MODEL_H);
    const data = new Float32Array(MODEL_C * MODEL_W * MODEL_H);
    fillArray(data, imageData);
    const inputTensor = new ort.Tensor('float32', data, [1, MODEL_C, MODEL_H, MODEL_W]);

    // Run inference
    const outputMap = await session.run({ images: inputTensor });
    const outputTensor = outputMap.output0;

    // Process detections and apply NMS
    const detections = processOutput(outputTensor);

    console.log(detections)
    return detections;
}


// ======================================================================== //
// DRAW                                                                     //
// ======================================================================== //

// Draw bounding boxes on the SVG overlay
function drawBoundingBoxes(detections) {
    const svg = document.getElementById('overlay');
    svg.innerHTML = ''; // Clear previous boxes
    const padding = (640 - 480) / 2; // 80
    detections.forEach(det => {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', -det.w / 2);
        rect.setAttribute('y', -det.h / 2);
        rect.setAttribute('width', det.w);
        rect.setAttribute('height', det.h);
        rect.setAttribute('transform', `translate(${det.cx}, ${det.cy - padding}) rotate(${det.theta * 180 / Math.PI})`);
        rect.setAttribute('stroke', 'red');
        rect.setAttribute('stroke-width', '2');
        rect.setAttribute('fill', 'none');
        svg.appendChild(rect);
    });
}

// ======================================================================== //
// MAIN                                                                     //
// ======================================================================== //

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
    const startCameraButton = document.getElementById('startCamera');

    // Load the ONNX model (assuming this is part of your original code)
    status.textContent = 'Loading model...';
    const session = await ort.InferenceSession.create('1ipho2mn_best.onnx');
    status.textContent = 'Model loaded. Click "Start Camera" to begin.';

    // Start camera on button click
    startCameraButton.addEventListener('click', async () => {
        startCameraButton.disabled = true;
        status.textContent = 'Requesting camera access...';
        try {
            const initialStream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = initialStream;
            currentStream = initialStream;
            const deviceId = initialStream.getVideoTracks()[0].getSettings().deviceId;
            status.textContent = 'Camera access granted. Populating devices...';
            await populateDevices(deviceId);
            status.textContent = 'Running detection...';
            inferenceLoop(session); // Assume this is defined elsewhere
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

// ======================================================================== //
// END                                                                      //
// ======================================================================== //
