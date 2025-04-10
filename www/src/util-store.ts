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
 * In Safari, device labels are only available after permission is granted.
 * This function handles that case by requesting camera access if needed.
 */
export async function populateDevices(): Promise<void> {
  try {
    // First try to enumerate devices
    let devices = await navigator.mediaDevices.enumerateDevices();
    let videoDevices = devices.filter(device => device.kind === 'videoinput');

    // Check if we have labels for the devices (Safari won't have them until permission is granted)
    const hasLabels = videoDevices.some(device => device.label);

    // If we don't have labels and we have video devices, we need to request permission
    if (!hasLabels && videoDevices.length > 0) {
      console.log('No device labels available. Requesting camera permission...');
      try {
        // Request camera access to get labels
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });

        // Stop all tracks immediately - we just needed this for permissions
        stream.getTracks().forEach(track => track.stop());

        // Now try again to get the devices with labels
        devices = await navigator.mediaDevices.enumerateDevices();
        videoDevices = devices.filter(device => device.kind === 'videoinput');
        console.log('Got device labels after permission:', videoDevices);
      } catch (permError) {
        console.error('Permission request failed:', permError);
        // Continue with the devices we have, even without labels
      }
    }

    $devices.set(videoDevices);
    return videoDevices;
  } catch (error) {
    console.error('Failed to populate devices:', error);
    $status.set('Error accessing devices.');
    return [];
  }
}
