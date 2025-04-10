import { atom } from 'nanostores';
import type {Detection, Device, Stats} from "./types";

export const $devices = atom<Device[]>([]);
export const $selectedDevice = atom<string|null>(null);
export const $isStreaming = atom<boolean>(false);
export const $detections = atom<Detection[]>([]);
export const $selectedId = atom<number|null>(null);
export const $status = atom<string>('Click "Start Streaming" to begin.');
export const $stats = atom<Stats>({messagesSent: 0, messagesReceived: 0, processTime: null});
export const $videoDimensions = atom<{width: number, height: number} | null>(null);
export const $wsConnected = atom<boolean>(false);


// overlay modes
export const $showOverlayPolygon = atom<boolean>(false);
export const $showOverlayPolygonClosed = atom<boolean>(false);
export const $showOverlayXyxyxyxy = atom<boolean>(true);

// send delay
export const $sendPeriodMs = atom<number>(1000/15);  // 15FPS
export const $sendQuality = atom<number>(0.5);


/**
 * Populates the $devices atom with available video input devices.
 *
 * Note: In Safari, device labels are only available after permission is granted.
 */
export async function populateDevices(): Promise<Device[]> {
  try {
    // Get all devices
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    // Update the store
    $devices.set(videoDevices);
    return videoDevices;
  } catch (error) {
    console.error('Failed to populate devices:', error);
    $status.set('Error accessing devices.');
    return [];
  }
}
